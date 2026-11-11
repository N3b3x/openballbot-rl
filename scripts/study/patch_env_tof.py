"""One-off patch: add ToF + privileged height-scan observations to ballbot_env.py (idempotent)."""
p = "ballbot_gym/envs/ballbot_env.py"
s = open(p, encoding="utf-8").read()
if "ToFArray" in s:
    raise SystemExit("already patched")


def rep(o, n):
    global s
    assert s.count(o) == 1, o[:60]
    s = s.replace(o, n, 1)


rep("from ballbot_gym.utils.contact_frame import ContactFrameStepper\n",
    "from ballbot_gym.utils.contact_frame import ContactFrameStepper\nfrom ballbot_gym.sensors.tof import ToFArray\n")
rep("        # Load MuJoCo model and create simulation data structure",
    '''        # Optional multizone ToF sensors (env_config["tof"] = list of sensor dicts) and the privileged
        # height scan for teacher policies (env_config["height_scan"] = {"x": [lo, hi, n], "y": [lo, hi, n]}).
        self._tof_cfg = env_config.get("tof") or []
        self._scan_cfg = env_config.get("height_scan")
        self._tof_extra_noise = float(env_settings.get("tof_extra_noise", 0.0))
        self._tof_extra_dropout = float(env_settings.get("tof_extra_dropout", 0.0))

        # Load MuJoCo model and create simulation data structure''')
rep("        self._stepper = ContactFrameStepper(self.model, fix_frames=self.fix_wheel_contacts)\n",
    '''        self._stepper = ContactFrameStepper(self.model, fix_frames=self.fix_wheel_contacts)
        self._tof = ToFArray(self.model, self._tof_cfg) if self._tof_cfg else None
        spaces = dict(self.observation_space.spaces)
        if self._tof is not None:
            spaces["tof"] = gym.spaces.Box(0.0, 1.0, shape=(self._tof.size,), dtype=_default_dtype)
        if self._scan_cfg:
            n_scan = int(self._scan_cfg["x"][2]) * int(self._scan_cfg["y"][2])
            spaces["height_scan"] = gym.spaces.Box(-1.0, 1.0, shape=(n_scan,), dtype=_default_dtype)
        self.observation_space = gym.spaces.Dict(spaces)
''')
rep("        if self.goal_mode:\n            if self.goal_mode == \"relative\":\n                g = np.clip",
    '''        if self._tof is not None:
            obs["tof"] = self._tof(self.data, self._np_random, self._tof_extra_noise, self._tof_extra_dropout)
        if self._scan_cfg:
            obs["height_scan"] = self._height_scan(position)

        if self.goal_mode:
            if self.goal_mode == "relative":
                g = np.clip''')
rep("    def _get_obs(self, last_ctrl):",
    '''    def _height_scan(self, position):
        """Terrain heights [m] relative to the terrain under the robot, on a grid around/ahead of it
        (privileged information for teacher policies), clipped to +-10 cm and scaled to [-1, 1]."""
        n = self.model.hfield_nrow.item()
        half = self.model.hfield_size[0, 0]
        z = self.model.hfield_data.reshape(n, n) * self.model.hfield_size[0, 2]

        def idx(v):
            return np.clip(np.round((v + half) / (2 * half) * (n - 1)).astype(int), 0, n - 1)

        xs = np.linspace(*self._scan_cfg["x"][:2], int(self._scan_cfg["x"][2])) + position[0]
        ys = np.linspace(*self._scan_cfg["y"][:2], int(self._scan_cfg["y"][2])) + position[1]
        ref = z[idx(np.array(position[0])), idx(np.array(position[1]))]
        h = z[np.ix_(idx(xs), idx(ys))]
        return np.clip((h - ref).ravel() / 0.1, -1.0, 1.0).astype(_default_dtype)

    def _get_obs(self, last_ctrl):''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
