"""Small rotation helpers (numpy, single robot). Quaternions are (w, x, y, z), scalar first, as in MuJoCo.

A quaternion q maps body vectors to world vectors: v_world = R(q) @ v_body.
"""
import numpy as np


def quat_to_mat(q):
    """(4,) quaternion (w, x, y, z) -> (3, 3) rotation matrix (body to world)."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R, q_prev=None):
    """(3, 3) rotation matrix -> unit quaternion (w, x, y, z).

    q and -q are the same rotation, but the policy observes the rotation vector computed from q (see rotvec), which differs between the
    two. In simulation the quaternion comes from continuous integration, so it never flips sign. Pass the previous quaternion to keep
    that continuity (the sign closest to q_prev is returned); without it the w >= 0 hemisphere is used.
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    if q_prev is not None:
        if np.dot(q, q_prev) < 0:
            q = -q
    elif q[0] < 0:
        q = -q
    return q


def rotvec(q):
    """Rotation vector of quaternion q exactly as the training environment computes it: axis * 2 atan2(|v|, w).
    Note: for w < 0 the angle exceeds pi (no wrapping), matching the simulator."""
    v = np.asarray(q[1:4], dtype=float)
    n = max(np.linalg.norm(v), 1e-9)
    return v / n * (2.0 * np.arctan2(n, q[0]))


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tilt_deg(q):
    """Angle between the body z axis and world z, degrees."""
    up_z = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
    return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def expm_so3(phi):
    """Rodrigues: exp of the skew matrix of phi (3,)."""
    th = np.linalg.norm(phi)
    if th < 1e-12:
        return np.eye(3) + skew(phi)
    K = skew(phi / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def orthonormalise(R):
    u, _, vt = np.linalg.svd(R)
    return u @ vt


def wrap_angle(a):
    """Wrap to [-pi, pi) the way torch.remainder does in the simulator."""
    return np.remainder(a + np.pi, 2 * np.pi) - np.pi
