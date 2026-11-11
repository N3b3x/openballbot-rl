"""State estimator node (thin wrapper over ballbot_control.core.estimator.StateEstimator) for Nav2 and the rest of the stack.

Subscribes   /imu           sensor_msgs/Imu         gyro (rad/s) and specific force (m/s^2), base_link frame
             /joint_states  sensor_msgs/JointState  wheel encoder rates (velocity field, else differentiated position)
Publishes    /odom          nav_msgs/Odometry       frame odom, child base_link; pose = base_link origin (ball centre + R r_origin),
                                                    twist in base_link (REP 105 / nav_msgs convention), covariance from the Kalman filter
             TF odom -> base_link (param publish_tf)
Runs at rate_hz (100 Hz, the filter's design rate) on the latest samples. The tracking controller runs its own instance of the same
estimator on the same messages (no extra transport latency in the balance loop); both see identical inputs and therefore agree.
Yaw has no absolute reference (no magnetometer or vision): it drifts with the residual gyro bias. Feed Nav2 a localisation source
(AMCL, SLAM) for map -> odom.
"""
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from tf2_ros import TransformBroadcaster

from ballbot_control.core import StateEstimator, load_bundle


class EstimatorNode(Node):
    def __init__(self):
        super().__init__("ballbot_estimator")
        p = self.declare_parameter
        p("bundle_path", ""); p("design", "nom"); p("rate_hz", 100.0); p("odom_frame", "odom"); p("base_frame", "base_link")
        p("publish_tf", True); p("wheel_joints", ["wheel_joint_0", "wheel_joint_1", "wheel_joint_2"]); p("sensor_timeout", 0.05)
        g = lambda n: self.get_parameter(n).value
        b = load_bundle(g("bundle_path"))
        kin = b.design(g("design"))["kinematics"]
        self.est = StateEstimator(dict(b["estimator"], track_point="ball_centre"), kin, b.dt)
        self.r_origin = np.asarray(kin["base_origin_from_ball_centre_body"])
        self.ball_r = float(kin["ball_radius"])
        self.odom_frame, self.base_frame = g("odom_frame"), g("base_frame")
        self.wheels, self.timeout = list(g("wheel_joints")), float(g("sensor_timeout"))
        self.tf = TransformBroadcaster(self) if g("publish_tf") else None
        self.gyro = self.acc = self.wr = None; self.t_imu = self.t_enc = None; self._q_prev = None; self._t_q_prev = None
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Imu, "imu", self._on_imu, qos)
        self.create_subscription(JointState, "joint_states", self._on_joints, qos)
        self.pub = self.create_publisher(Odometry, "odom", 10)
        self.t0 = time.monotonic()
        self.create_timer(1.0 / float(g("rate_hz")), self._tick)

    def _now(self):
        return time.monotonic() - self.t0

    def _on_imu(self, m):
        self.gyro = np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
        self.acc = np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        self.t_imu = self._now()

    def _on_joints(self, m):
        try:
            idx = [m.name.index(n) for n in self.wheels]
        except ValueError:
            return
        now = self._now()
        if len(m.velocity) == len(m.name):
            self.wr = np.array([m.velocity[i] for i in idx])
        elif len(m.position) == len(m.name):
            q = np.array([m.position[i] for i in idx])
            if self._q_prev is not None and now > self._t_q_prev:
                self.wr = (q - self._q_prev) / (now - self._t_q_prev)
            self._q_prev, self._t_q_prev = q, now
        self.t_enc = now

    def _tick(self):
        now = self._now()
        if self.gyro is None or self.wr is None or now - min(self.t_imu, self.t_enc) > self.timeout:
            return      # no fresh data: publish nothing rather than a stale estimate
        s, info = self.est.step(self.wr, self.gyro, self.acc)
        R = info["R"]
        pos = np.array([info["pos"][0], info["pos"][1], self.ball_r]) + R @ self.r_origin     # base_link origin
        stamp = self.get_clock().now().to_msg()
        o = Odometry(); o.header.stamp = stamp; o.header.frame_id = self.odom_frame; o.child_frame_id = self.base_frame
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = (float(v) for v in pos)
        w, x, y, z = (float(v) for v in s.quat)
        o.pose.pose.orientation.x, o.pose.pose.orientation.y, o.pose.pose.orientation.z, o.pose.pose.orientation.w = x, y, z, w
        P = info["P"]
        cov = np.zeros((6, 6)); cov[0:2, 0:2] = P[0:2, 0:2]; cov[2, 2] = 1e-4; cov[3, 3] = cov[4, 4] = 1e-4; cov[5, 5] = 1e-2
        o.pose.covariance = [float(v) for v in cov.ravel()]
        v_body = R.T @ s.v_origin
        o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = (float(v) for v in v_body)
        o.twist.twist.angular.x, o.twist.twist.angular.y, o.twist.twist.angular.z = (float(v) for v in s.omega_body)
        tc = np.zeros((6, 6)); tc[0:2, 0:2] = P[2:4, 2:4]; tc[2, 2] = 1e-2; tc[3, 3] = tc[4, 4] = tc[5, 5] = 1e-4
        o.twist.covariance = [float(v) for v in tc.ravel()]
        self.pub.publish(o)
        if self.tf is not None:
            t = TransformStamped(); t.header.stamp = stamp; t.header.frame_id = self.odom_frame; t.child_frame_id = self.base_frame
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in pos)
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = x, y, z, w
            self.tf.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = EstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
