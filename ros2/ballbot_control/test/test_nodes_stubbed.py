"""The ROS nodes against stand-in rclpy / message modules (no ROS on this machine). This checks the wiring (parameters, topics,
callbacks, services, the tick) and that the nodes stay thin; it says nothing about real ROS behaviour (QoS, executors, timing)."""
import ast
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1] / "ballbot_control"


def test_core_has_no_ros_imports():
    for f in (PKG / "core").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for n in names:
                assert not n.split(".")[0] in ("rclpy", "std_msgs", "sensor_msgs", "geometry_msgs", "nav_msgs", "tf2_ros"), f"{f.name}: {n}"


class _Msg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _vec():
    return _Msg(x=0.0, y=0.0, z=0.0)


class Imu(_Msg):
    def __init__(self):
        super().__init__(angular_velocity=_vec(), linear_acceleration=_vec())


class JointState(_Msg):
    def __init__(self, name=(), position=(), velocity=()):
        super().__init__(name=list(name), position=list(position), velocity=list(velocity))


class Twist(_Msg):
    def __init__(self):
        super().__init__(linear=_vec(), angular=_vec())


class _Header(_Msg):
    def __init__(self):
        super().__init__(stamp=None, frame_id="")


class _Diag(_Msg):
    OK, WARN, ERROR = 0, 1, 2

    def __init__(self, **kw):
        super().__init__(level=0, message="", values=[], **kw)


class _DiagArray(_Msg):
    def __init__(self, status=()):
        super().__init__(status=list(status), header=_Header())


class FakeNode:
    instances = []

    def __init__(self, name):
        self.name, self.params, self.subs, self.pubs, self.srvs, self.timers = name, {}, {}, {}, {}, []
        FakeNode.instances.append(self)

    def declare_parameter(self, n, v):
        self.params.setdefault(n, v)

    def get_parameter(self, n):
        return _Msg(value=self.params[n])

    def create_subscription(self, typ, topic, cb, qos):
        self.subs[topic] = cb

    def create_publisher(self, typ, topic, qos):
        pub = _Msg(sent=[]); pub.publish = pub.sent.append
        self.pubs[topic] = pub
        return pub

    def create_service(self, typ, name, cb):
        self.srvs[name] = cb

    def create_timer(self, period, cb):
        self.timers.append((period, cb))

    def get_logger(self):
        return _Msg(info=lambda *a: None, warn=lambda *a: None)

    def get_clock(self):
        return _Msg(now=lambda: _Msg(to_msg=lambda: None))


@pytest.fixture
def stub_ros(monkeypatch, bundle):
    mods = {
        "rclpy": types.ModuleType("rclpy"), "rclpy.node": types.ModuleType("rclpy.node"), "rclpy.qos": types.ModuleType("rclpy.qos"),
        "sensor_msgs": types.ModuleType("sensor_msgs"), "sensor_msgs.msg": types.ModuleType("sensor_msgs.msg"),
        "geometry_msgs": types.ModuleType("geometry_msgs"), "geometry_msgs.msg": types.ModuleType("geometry_msgs.msg"),
        "std_msgs": types.ModuleType("std_msgs"), "std_msgs.msg": types.ModuleType("std_msgs.msg"),
        "std_srvs": types.ModuleType("std_srvs"), "std_srvs.srv": types.ModuleType("std_srvs.srv"),
        "diagnostic_msgs": types.ModuleType("diagnostic_msgs"), "diagnostic_msgs.msg": types.ModuleType("diagnostic_msgs.msg"),
    }
    mods["rclpy.node"].Node = FakeNode
    q = mods["rclpy.qos"]; q.QoSProfile = lambda **kw: kw; q.ReliabilityPolicy = _Msg(BEST_EFFORT=1); q.HistoryPolicy = _Msg(KEEP_LAST=1)
    mods["sensor_msgs.msg"].Imu, mods["sensor_msgs.msg"].JointState = Imu, JointState
    mods["geometry_msgs.msg"].Twist, mods["geometry_msgs.msg"].PoseStamped = Twist, _Msg
    mods["std_msgs.msg"].Float64MultiArray = lambda data: _Msg(data=list(data))
    mods["std_srvs.srv"].SetBool, mods["std_srvs.srv"].Trigger = _Msg, _Msg
    d = mods["diagnostic_msgs.msg"]; d.DiagnosticArray, d.DiagnosticStatus, d.KeyValue = _DiagArray, _Diag, _Msg
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    monkeypatch.delitem(sys.modules, "ballbot_control.nodes.controller_node", raising=False)
    FakeNode.instances.clear()
    yield


def test_controller_node_wiring(stub_ros, monkeypatch):
    from parity_replay import DEFAULT_BUNDLE
    from ballbot_control.nodes import controller_node as cn
    orig = FakeNode.__init__

    def init(self, name):
        orig(self, name); self.params["bundle_path"] = str(DEFAULT_BUNDLE)
    monkeypatch.setattr(FakeNode, "__init__", init)
    node = cn.ControllerNode()
    assert set(node.subs) == {"imu", "joint_states", "cmd_vel"}
    assert {"wheel_torque_cmd", "ballbot/diagnostics"} <= set(node.pubs)
    period, tick = node.timers[0]
    assert period == pytest.approx(0.01)
    imu = Imu(); imu.linear_acceleration.z = 9.81
    js = JointState(name=["other", "wheel_joint_0", "wheel_joint_1", "wheel_joint_2"], position=[0, 0, 0, 0], velocity=[0, 0, 0, 0])
    for _ in range(3):
        node.subs["imu"](imu); node.subs["joint_states"](js); tick()
    assert node.loop.sup.mode == "HOLD" and node.pubs["wheel_torque_cmd"].sent[-1].data == [0.0, 0.0, 0.0]
    res = node.srvs["ballbot/enable"](_Msg(data=True), _Msg())
    assert res.success
    tw = Twist(); tw.linear.x = 0.3
    node.subs["cmd_vel"](tw)
    for _ in range(10):
        node.subs["imu"](imu); node.subs["joint_states"](js); tick()
    assert node.loop.sup.mode == "ACTIVE"
    assert len(node.pubs["wheel_torque_cmd"].sent[-1].data) == 3
    assert node.loop.ctrl.mref.s > 0          # the command reached the motion layer
    node.srvs["ballbot/enable"](_Msg(data=False), _Msg()); tick()
    assert node.pubs["wheel_torque_cmd"].sent[-1].data == [0.0, 0.0, 0.0]
    assert any(m.status for m in node.pubs["ballbot/diagnostics"].sent) or node.diag_every > 13
