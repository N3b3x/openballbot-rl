"""One-off patch: velocity-command tracking mode (any direction + yaw rate) and configurable limits."""
p = "ballbot_gym/envs/ballbot_env.py"
s = open(p, encoding="utf-8").read()
if "command_mode" in s:
    raise SystemExit("already patched")


def rep(o, n):
    global s
    assert s.count(o) == 1, o[:70]
    s = s.replace(o, n, 1)


rep("        # Re-align omniwheel contact frames in Python",
    '''        # Velocity-command mode: obs["cmd"] = [vx, vy, wz] (world frame, m/s and rad/s), resampled every
        # cmd_resample_steps decisions; reward = exp tracking of planar velocity and yaw rate.
        self.command_mode = bool(env_settings.get("command_mode", False))
        self.cmd_vmax = float(env_settings.get("cmd_vmax", 1.0))
        self.cmd_wmax = float(env_settings.get("cmd_wmax", 2.0))
        self.cmd_resample_steps = int(env_settings.get("cmd_resample_steps", 200))
        self.cmd_zero_prob = float(env_settings.get("cmd_zero_prob", 0.1))
        self.cmd_sigma_v = float(env_settings.get("cmd_sigma_v", 0.25))
        self.cmd_sigma_w = float(env_settings.get("cmd_sigma_w", 0.5))
        self.cmd_fixed = env_settings.get("cmd_fixed")  # optional fixed [vx, vy, wz] for evaluation
        self.wheel_damping = env_settings.get("wheel_damping")  # override joint damping [N m s/rad]
        self.obs_vel_clip = float(env_settings.get("obs_vel_clip", 2.0))
        self.cmd = np.zeros(3)
        self.cmd_err_hist = []
        # Re-align omniwheel contact frames in Python''')
rep('''            spaces["goal"] = gym.spaces.Box(-1.5, 1.5, shape=(2, ), dtype=_default_dtype)
            self.observation_space = gym.spaces.Dict(spaces)
''', '''            spaces["goal"] = gym.spaces.Box(-1.5, 1.5, shape=(2, ), dtype=_default_dtype)
            self.observation_space = gym.spaces.Dict(spaces)
        if self.command_mode:
            spaces = dict(self.observation_space.spaces)
            spaces["cmd"] = gym.spaces.Box(-5.0, 5.0, shape=(3, ), dtype=_default_dtype)
            spaces["heading"] = gym.spaces.Box(-1.0, 1.0, shape=(2, ), dtype=_default_dtype)
            self.observation_space = gym.spaces.Dict(spaces)
''')
rep("        self._stepper = ContactFrameStepper(self.model, fix_frames=self.fix_wheel_contacts)\n        self._tof",
    '''        self._stepper = ContactFrameStepper(self.model, fix_frames=self.fix_wheel_contacts)
        # The actuators are torque motors: "max_wheel_velocity" scales the commanded torque [N m]
        # (the XML hard-codes +-10), so widen the control range when asking for more torque.
        self.model.actuator_ctrlrange[:] = np.array([-1.0, 1.0]) * max(self.max_wheel_velocity, 1e-6)
        if self.wheel_damping is not None:
            for k in range(3):
                self.model.dof_damping[self.model.joint(f"wheel_joint_{k}").dofadr[0]] = float(self.wheel_damping)
        self._tof''')
rep("        vel = np.clip(linear_vel, a_min=-2.0, a_max=2.0)\n        angular_vel = np.clip(angular_vel, a_min=-2.0, a_max=2.0)",
    "        vel = np.clip(linear_vel, a_min=-self.obs_vel_clip, a_max=self.obs_vel_clip)\n        angular_vel = np.clip(angular_vel, a_min=-self.obs_vel_clip, a_max=self.obs_vel_clip)")
rep('''        if self.goal_mode:
            if self.goal_mode == "relative":
                g = np.clip''', '''        if self.command_mode:
            obs["cmd"] = self.cmd.astype(_default_dtype)
            obs["heading"] = self._heading_sincos().astype(_default_dtype)

        if self.goal_mode:
            if self.goal_mode == "relative":
                g = np.clip''')
rep("    def _height_scan(self, position):",
    '''    def _yaw(self):
        w, x, y, z = self.data.xquat[self.model.body("base").id]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def _heading_sincos(self):
        yaw = self._yaw()
        return np.array([np.sin(yaw), np.cos(yaw)])

    def _sample_cmd(self):
        if self.cmd_fixed is not None:
            self.cmd = np.asarray(self.cmd_fixed, dtype=float)
            return
        rng = self._np_random
        if rng.random() < self.cmd_zero_prob:
            self.cmd = np.zeros(3)
            return
        sp = rng.uniform(0.0, self.cmd_vmax)
        ang = rng.uniform(0.0, 2 * np.pi)
        self.cmd = np.array([sp * np.cos(ang), sp * np.sin(ang), rng.uniform(-self.cmd_wmax, self.cmd_wmax)])

    def _height_scan(self, position):''')
rep("        if self.goal_mode:\n            phi = np.radians",
    "        if self.command_mode:\n            self._sample_cmd()\n        if self.goal_mode:\n            phi = np.radians")
rep('''        if self.goal_mode:
            if self.goal_mode == "relative":
                vec = self.goal_xy - info["pos2d"]''', '''        if self.command_mode:
            if self.step_counter > 0 and self.step_counter % self.cmd_resample_steps == 0:
                self._sample_cmd()
                obs["cmd"] = self.cmd.astype(_default_dtype)
            ev = float(np.linalg.norm(obs["vel"][:2] - self.cmd[:2]))
            ew = float(obs["angular_vel"][2] - self.cmd[2])
            raw_reward = float(np.exp(-ev ** 2 / self.cmd_sigma_v ** 2) + np.exp(-ew ** 2 / self.cmd_sigma_w ** 2))
            info["vel_err"], info["yaw_err"] = ev, ew
            info["speed"] = float(np.linalg.norm(obs["vel"][:2]))
            info["yaw_rate"] = float(obs["angular_vel"][2])
            info["cmd"] = self.cmd.copy()
        elif self.goal_mode:
            if self.goal_mode == "relative":
                vec = self.goal_xy - info["pos2d"]''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
