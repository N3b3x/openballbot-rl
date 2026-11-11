"""How well can the ballbot track its own position from its encoders and IMU? (simulation study, offline on collect_odom.py logs)

Truth: the ball centre (log ball_p is the ball BODY origin; the centre is ball_p + R_ball @ ball_geom_offset). The estimator
(odometry.BallbotOdometry) starts at the true pose and runs on simulated sensors made from the noiseless log:
  encoders  wheel angle quantised to `cpr` counts per wheel revolution, rate = count difference / dt
  gyro      true body rate + constant bias (random per robot, sd bias_g) + random-walk bias + white noise
  accel     true specific force in the body frame (from the base velocity, differenced) + bias + white noise
Calibration C is fitted on half of the flat worlds and tested on the other half and on the terrain log.
Reported per sensor model: position error at 10 / 30 / 60 s (median and p90 over worlds still upright), error as a percentage of
distance travelled, and yaw error.
  python eval_odom.py --cal odom_flat --test odom_flat odom_tiles
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from odometry import BallbotOdometry, calibrate
from scipy.spatial.transform import Rotation as Rot
ODOM = Path(__file__).resolve().parents[2] / "outputs" / "study" / "warp_ppo" / "odom"
ap = argparse.ArgumentParser()
ap.add_argument("--cal", default="odom_flat"); ap.add_argument("--test", nargs="+", default=["odom_flat", "odom_tiles"])
ap.add_argument("--out", default="odom_results")
a = ap.parse_args()


def load(name):
    d = dict(np.load(ODOM / f"{name}.npz")); meta = json.loads(str(d.pop("meta")))
    return d, meta


def centre(d, meta):
    T_, N_ = d["ball_p"].shape[:2]
    Rk = Rot.from_quat(d["ball_quat"].reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix()
    return d["ball_p"] + np.einsum("nij,j->ni", Rk, np.array(meta.get("ball_geom_offset", [0, 0, -0.14]))).reshape(T_, N_, 3)


def sensors(d, model, rng):
    T_, N_ = d["gyro"].shape[:2]; dt = 0.01
    cpr = model["cpr"]
    q = d["wheel_q"].astype(np.float64)
    if cpr:
        q = np.round(q * cpr / (2 * np.pi)) * 2 * np.pi / cpr
    wr = np.zeros_like(q); wr[1:] = (q[1:] - q[:-1]) / dt
    if not cpr:
        wr = d["wheel_w"].astype(np.float64)
    bias0 = rng.normal(0, model["bias_g"], (N_, 3))
    rw = np.cumsum(rng.normal(0, model["rw_g"] * np.sqrt(dt), (T_, N_, 3)), 0)
    gyro = d["gyro"] + bias0[None] + rw + rng.normal(0, model["noise_g"], (T_, N_, 3))
    Rb = Rot.from_quat(d["quat"].reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(T_, N_, 3, 3)
    aw = np.zeros_like(d["base_v"], dtype=np.float64); aw[1:] = (d["base_v"][1:] - d["base_v"][:-1]) / dt
    f_body = np.einsum("tnji,tnj->tni", Rb, aw - np.array([0, 0, -9.81]))
    acc = f_body + rng.normal(0, model["bias_a"], (N_, 3))[None] + rng.normal(0, model["noise_a"], (T_, N_, 3))
    return wr, gyro, acc, Rb


MODELS = {
    "ideal (true rates, no noise)": dict(cpr=0, bias_g=0, rw_g=0, noise_g=0, bias_a=0, noise_a=0, r_err=0.0),
    "realistic MEMS IMU + 4096-count encoders": dict(cpr=4096, bias_g=0.005, rw_g=0.0005, noise_g=0.003, bias_a=0.03, noise_a=0.05, r_err=0.0),
    "realistic + ball radius 2 percent off": dict(cpr=4096, bias_g=0.005, rw_g=0.0005, noise_g=0.003, bias_a=0.03, noise_a=0.05, r_err=0.02),
    "realistic, uncalibrated (planar model W)": dict(cpr=4096, bias_g=0.005, rw_g=0.0005, noise_g=0.003, bias_a=0.03, noise_a=0.05, r_err=0.0, uncal=True),
}
dcal, mcal = load(a.cal)
N_cal = dcal["alive"].shape[1]
C = calibrate(dcal, np.arange(N_cal // 2))
W = np.array(json.load(open(Path(__file__).resolve().parents[2] / "ballbot_gym" / "models" / "variants" / "dz_nom.json"))["params"]["W"])
C0 = np.linalg.inv(W)
print("calibration C =\n", np.round(C, 4), "\nplanar-model W^-1 =\n", np.round(C0, 4))
results = {}
for name in a.test:
    d, meta = load(name)
    T_, N_ = d["alive"].shape
    worlds = np.arange(N_ // 2, N_) if name == a.cal else np.arange(N_)
    d = {k: v[:, worlds] for k, v in d.items()}
    pc = centre(d, meta)[..., :2]
    alive = d["alive"]
    life = alive.sum(0)
    for mname, model in MODELS.items():
        rng = np.random.default_rng(0)
        wr, gyro, acc, Rb = sensors(d, model, rng)
        est = BallbotOdometry(len(worlds), C0 if model.get("uncal") else C, meta["ball_r"] * (1 + model["r_err"]), R0=Rb[0], p0=pc[0])
        P = np.zeros((T_, len(worlds), 2)); V = np.zeros_like(P); yawe = np.zeros((T_, len(worlds)))
        ytrue = np.arctan2(Rb[:, :, 1, 0], Rb[:, :, 0, 0])
        for t in range(T_):
            P[t], V[t], _ = est.step(wr[t], gyro[t], acc[t])
            yawe[t] = np.angle(np.exp(1j * (est.yaw() - ytrue[t])))
        err = np.linalg.norm(P - pc, axis=2)
        dist = np.concatenate([np.zeros((1, len(worlds))), np.cumsum(np.linalg.norm(np.diff(pc, axis=0), axis=2), 0)], 0)
        row = {}
        for sec in (10, 30, 60):
            t = min(sec * 100, T_) - 1
            ok = life > t
            if ok.sum() < 3:
                continue
            row[f"err_{sec}s_median_m"] = float(np.median(err[t, ok])); row[f"err_{sec}s_p90_m"] = float(np.percentile(err[t, ok], 90))
            row[f"pct_of_distance_{sec}s_median"] = float(np.median(100 * err[t, ok] / np.maximum(dist[t, ok], 1e-3)))
            row[f"dist_{sec}s_median_m"] = float(np.median(dist[t, ok])); row[f"yaw_err_{sec}s_median_deg"] = float(np.degrees(np.median(np.abs(yawe[t, ok])))); row[f"n_{sec}s"] = int(ok.sum())
        results[f"{name} | {mname}"] = row
        print(name, "|", mname, {k: round(v, 3) for k, v in row.items() if not k.startswith("n_")}, flush=True)
(ODOM / f"{a.out}.json").write_text(json.dumps(dict(cal=a.cal, C=C.tolist(), models=MODELS, results=results), indent=1))
