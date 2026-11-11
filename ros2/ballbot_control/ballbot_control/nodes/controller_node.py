"""Tracking controller node (thin wrapper over ballbot_control.core.runtime.ControlLoop).

Subscribes   /cmd_vel        geometry_msgs/Twist        desired velocity (Nav2 controller output; base_link frame by default)
             /goal_pose      geometry_msgs/PoseStamped  optional position goal in odom (param use_goal_topic); a /cmd_vel cancels it
             /imu            sensor_msgs/Imu            angular_velocity (rad/s) and linear_acceleration (specific force, m/s^2), base_link
             /joint_states   sensor_msgs/JointState     wheel encoder rates of the joints named in wheel_joints (velocity, else
                                                        differentiated position)
Publishes    /wheel_torque_cmd   std_msgs/Float64MultiArray  three wheel torques, N m, order of wheel_joints, every tick (zeros unless ACTIVE)
             /ballbot/diagnostics diagnostic_msgs/DiagnosticArray  mode, reason, tilt, residual authority, loop timing (at diag_rate_hz)
             /ballbot/debug       std_msgs/Float64MultiArray  [cmd(3), applied(3), lqr(3), residual(3), rs] (param publish_debug)
Services     /ballbot/enable      std_srvs/SetBool   true: request ACTIVE (granted only when upright and the state is fresh); false: HOLD
             /ballbot/clear_fault std_srvs/Trigger   FAULT -> HOLD
The control tick runs on a wall timer at control_rate_hz (100 Hz, the rate the policy was trained at). Sensor callbacks only store
the latest sample; the tick consumes the newest IMU and encoder samples (zero-order hold).
"""
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool, Trigger

from ballbot_control.core import ACTIVE, FAULT, load_bundle
from ballbot_control.core.runtime import ControlLoop, LoopConfig


class ControllerNode(Node):
    def __init__(self):
        super().__init__("ballbot_controller")
        p = self.declare_parameter
        p("bundle_path", ""); p("design", "nom"); p("policy_backend", "numpy"); p("residual_enabled", True)
        p("control_rate_hz", 100.0); p("cmd_timeout", 0.5); p("hold_on_timeout", True); p("cmd_vel_frame", "base_link")
        p("max_speed", 0.6); p("max_yaw_rate", 1.0); p("tilt_cutoff_deg", 30.0); p("engage_tilt_deg", 3.0)
        p("state_timeout", 0.05); p("sensor_timeout", 0.05); p("torque_limit", 0.0); p("track_point", "")
        p("wheel_joints", ["wheel_joint_0", "wheel_joint_1", "wheel_joint_2"]); p("auto_enable", False)
        p("use_goal_topic", False); p("publish_debug", False); p("diag_rate_hz", 5.0)
        g = lambda n: self.get_parameter(n).value
        bundle_path = g("bundle_path")
        if not bundle_path:
            raise RuntimeError("parameter bundle_path is required (bundle.json written by export_policy.py)")
        self.bundle = load_bundle(bundle_path)
        rate = float(g("control_rate_hz"))
        if abs(rate * self.bundle.dt - 1.0) > 1e-6:
            self.get_logger().warn(f"control_rate_hz {rate} differs from the bundle's training rate {1.0 / self.bundle.dt} Hz")
        cfg = LoopConfig(design=g("design"), policy_backend=g("policy_backend"), residual=bool(g("residual_enabled")),
                         cmd_timeout=g("cmd_timeout"), hold_on_timeout=g("hold_on_timeout"), cmd_vel_frame=g("cmd_vel_frame"),
                         max_speed=g("max_speed"), max_yaw_rate=g("max_yaw_rate"), tilt_cutoff_deg=g("tilt_cutoff_deg"),
                         engage_tilt_deg=g("engage_tilt_deg"), state_timeout=g("state_timeout"), sensor_timeout=g("sensor_timeout"),
                         torque_limit=g("torque_limit"), track_point=g("track_point") or None)
        self.loop = ControlLoop(self.bundle, cfg)
        self.wheels = list(g("wheel_joints"))
        self.publish_debug = bool(g("publish_debug"))
        if g("auto_enable"):
            self.loop.enable()

        self.gyro = self.acc = None; self.t_imu = None
        self.wr = None; self.t_enc = None; self._q_prev = None; self._t_q_prev = None
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Imu, "imu", self._on_imu, sensor_qos)
        self.create_subscription(JointState, "joint_states", self._on_joints, sensor_qos)
        self.create_subscription(Twist, "cmd_vel", self._on_cmd, 10)
        if g("use_goal_topic"):
            self.create_subscription(PoseStamped, "goal_pose", self._on_goal, 10)
        self.pub_tau = self.create_publisher(Float64MultiArray, "wheel_torque_cmd", 1)
        self.pub_diag = self.create_publisher(DiagnosticArray, "ballbot/diagnostics", 1)
        self.pub_dbg = self.create_publisher(Float64MultiArray, "ballbot/debug", 1) if self.publish_debug else None
        self.create_service(SetBool, "ballbot/enable", self._srv_enable)
        self.create_service(Trigger, "ballbot/clear_fault", self._srv_clear)
        self.t0 = time.monotonic()
        self.last = None; self.overruns = 0; self.max_dt = 0.0; self.tick_n = 0
        self.diag_every = max(1, int(round(rate / float(g("diag_rate_hz")))))
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"ballbot controller: bundle {self.bundle['source']['run']}, design {self.loop.ctrl.design_name}, "
                               f"{rate:g} Hz, goal point {self.loop.est.track}, startup mode HOLD (call /ballbot/enable)")

    def _now(self):
        return time.monotonic() - self.t0

    # ---- inputs
    def _on_imu(self, m):
        self.gyro = np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
        self.acc = np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        self.t_imu = self._now()

    def _on_joints(self, m):
        try:
            idx = [m.name.index(n) for n in self.wheels]
        except ValueError:
            return      # a joint_states message without the wheel joints (e.g. another publisher)
        now = self._now()
        if len(m.velocity) == len(m.name):
            self.wr = np.array([m.velocity[i] for i in idx])
        elif len(m.position) == len(m.name):      # encoder angles only: finite difference
            q = np.array([m.position[i] for i in idx])
            if self._q_prev is not None and now > self._t_q_prev:
                self.wr = (q - self._q_prev) / (now - self._t_q_prev)
            self._q_prev, self._t_q_prev = q, now
        self.t_enc = now

    def _on_cmd(self, m):
        self.loop.on_cmd_vel(self._now(), m.linear.x, m.linear.y, m.angular.z)

    def _on_goal(self, m):
        self.loop.on_goal(np.array([m.pose.position.x, m.pose.position.y]))

    def _srv_enable(self, req, res):
        if req.data:
            self.loop.enable(); res.message = "enable requested (granted when upright and the state is fresh)"
        else:
            self.loop.disable(); res.message = "disabled: HOLD"
        res.success = True
        return res

    def _srv_clear(self, req, res):
        self.loop.sup.clear_fault(); res.success = True; res.message = f"mode {self.loop.sup.mode}"
        return res

    # ---- 100 Hz tick
    def _tick(self):
        now = self._now()
        if self.last is not None:
            dt = now - self.last
            self.max_dt = max(self.max_dt, dt)
            if dt > 1.5 * self.bundle.dt:
                self.overruns += 1
        self.last = now
        have = self.gyro is not None and self.wr is not None
        t_sensor = min(self.t_imu, self.t_enc) if have else None
        r = self.loop.tick(now, self.gyro if have else np.zeros(3), self.acc if have else np.zeros(3), self.wr if have else np.zeros(3), t_sensor)
        self.pub_tau.publish(Float64MultiArray(data=[float(v) for v in r.torque]))
        if self.pub_dbg is not None and r.step is not None:
            s = r.step
            self.pub_dbg.publish(Float64MultiArray(data=[float(v) for v in np.concatenate([s.cmd, s.applied, s.lqr, s.action, [s.rs]])]))
        self.tick_n += 1
        if self.tick_n % self.diag_every == 0:
            self._publish_diag(r)

    def _publish_diag(self, r):
        st = DiagnosticStatus(name="ballbot_controller", hardware_id="ballbot")
        st.level = DiagnosticStatus.OK if r.mode == ACTIVE else (DiagnosticStatus.ERROR if r.mode == FAULT else DiagnosticStatus.WARN)
        st.message = f"{r.mode}: {r.reason}"
        kv = dict(mode=r.mode, reason=r.reason, overruns=self.overruns, max_tick_interval_s=f"{self.max_dt:.4f}",
                  torque=",".join(f"{v:.3f}" for v in r.torque))
        if r.state is not None:
            from ballbot_control.core.math3d import tilt_deg
            kv["tilt_deg"] = f"{tilt_deg(r.state.quat):.2f}"
        if r.step is not None:
            kv.update(rs=f"{r.step.rs:.3f}", cmd=",".join(f"{v:.3f}" for v in r.step.cmd), goal_on=str(r.step.goal_on))
        st.values = [KeyValue(key=k, value=str(v)) for k, v in kv.items()]
        msg = DiagnosticArray(status=[st]); msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_diag.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)      # single-threaded executor: callbacks never run concurrently with the tick
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_tau.publish(Float64MultiArray(data=[0.0, 0.0, 0.0]))      # best effort: leave the motors at zero torque
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
