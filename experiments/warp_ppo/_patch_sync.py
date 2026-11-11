"""One-off patch: make train_dev.py reset / command sampling / pushes free of host syncs."""
p = 'train_dev.py'
s = open(p).read()

a = s.index("    def sample_cmd(self, idx):")
b = s.index("    def goal_feat(self):")
new = '''    def sample_cmd(self, mask):
        """Resample the command (and goal mode) of the worlds in mask (bool over N). Computed for all worlds and selected with where: no host sync."""
        N = self.N
        sp, ang, wz = self.U(N, hi=self.cmd_vmax) * self.f_w, self.U(N, hi=2 * np.pi), self.U(N, lo=-self.cmd_wmax, hi=self.cmd_wmax)
        c = torch.stack([sp * torch.cos(ang), sp * torch.sin(ang), wz], 1)
        c = torch.where((self.U(N) < self.zero_p)[:, None], torch.zeros_like(c), c)
        if self.edge_p > 0:      # edge-focused: with prob edge_p drive across the ridge (outward while |y| < 1.8 m, back inward beyond)
            y = self.qpos[:, 11]
            out = torch.where(y.abs() < 0.02, torch.where(self.U(N) < 0.5, -1.0, 1.0), torch.sign(y))
            tow = torch.where(y.abs() < 1.8, out, -out)
            ang_e = tow * (np.pi / 2) + self.U(N, lo=-0.6, hi=0.6)
            spd = self.U(N, lo=0.15, hi=self.cmd_vmax) * self.f_w
            m_e = (self.U(N) < self.edge_p)[:, None]
            ce = torch.stack([spd * torch.cos(ang_e), spd * torch.sin(ang_e), c[:, 2]], 1)
            c = torch.where(m_e, ce, c)
        self.cmd = torch.where(mask[:, None], c, self.cmd)
        if self.goal_p > 0:
            gm = self.U(N) < self.goal_p
            self.goal_mode = torch.where(mask, gm, self.goal_mode)
            self.path_left = torch.where(mask, torch.randint(0, 4, (N,), device=DEV, generator=self.g), self.path_left)
            self.new_goal(mask & gm)

    def new_goal(self, mask):
        """Draw a goal 0.7 to 1.8 m away (clamped to the tile square) for the worlds in mask; computed for all worlds, no sync."""
        N = self.N
        ang = self.U(N, hi=2 * np.pi); r = self.U(N, lo=0.7, hi=1.8) if self.tiles else self.U(N, lo=1.0, hi=3.0)
        gx = self.qpos[:, 10] + r * torch.cos(ang); gy = self.qpos[:, 11] + r * torch.sin(ang)
        if self.tiles:
            c = self.tile_xy[self.cur_tile]
            gx = torch.minimum(torch.maximum(gx, c[:, 0] - 1.3), c[:, 0] + 1.3); gy = torch.minimum(torch.maximum(gy, c[:, 1] - 1.3), c[:, 1] + 1.3)
        g = torch.stack([gx, gy], 1)
        self.goal = torch.where(mask[:, None], g, self.goal)

    def update_goal_cmd(self):
        """Goal-directed worlds: the velocity command is a saturating proportional rule on the goal error (speed free up to cmd_vmax), chained waypoints on arrival."""
        if self.goal_p <= 0:
            return
        e = self.goal - self.qpos[:, 10:12]
        d = e.norm(dim=1)
        arrived = self.goal_mode & (d < 0.2) & (self.path_left > 0)
        self.path_left = self.path_left - arrived.long()
        self.new_goal(arrived)
        e = self.goal - self.qpos[:, 10:12]; d = e.norm(dim=1)
        v = e / d.clamp_min(1e-6)[:, None] * torch.minimum(1.5 * d, self.cmd_vmax * self.f_w)[:, None]
        v = torch.where((d < 0.05)[:, None], torch.zeros_like(v), v)
        cx = torch.where(self.goal_mode[:, None], v, self.cmd[:, :2])
        cz = torch.where(self.goal_mode, torch.zeros_like(self.cmd[:, 2]), self.cmd[:, 2])
        self.cmd = torch.cat([cx, cz[:, None]], 1)

'''
s = s[:a] + new + s[b:]


def rep(old, new_, count=1):
    global s
    assert old in s, old[:60]
    s = s.replace(old, new_, count)


rep('''    def reset(self, idx):
        n = int(idx.sum())
        if n == 0:
            return
        ii0 = idx.nonzero()[:, 0]''', '''    def reset(self, idx):
        ii0 = idx.nonzero()[:, 0]      # the only host sync in reset
        n = ii0.numel()
        if n == 0:
            return''')
rep("            self.cur_tile[idx] = tid\n", "            self.cur_tile[ii0] = tid\n")
rep('''        self.qpos[idx] = q
        self.qvel[idx] = 0
        self.warm[idx] = 0
        self.ctrl[idx] = 0
        if self.motor:
            self.tau_act[idx] = 0
        self.xfrc[idx] = 0
        self.sample_cmd(idx)
        self.update_goal_cmd()
        self.vint[idx] = 0; self.last_a[idx] = 0; self.has_prev[idx] = False; self.steps[idx] = 0
        self.tq[idx] = 1 + (self.U(n, lo=-0.15, hi=0.15) if self.dr else 0.0)''', '''        self.qpos[ii0] = q
        self.qvel[ii0] = 0
        self.warm[ii0] = 0
        self.ctrl[ii0] = 0
        if self.motor:
            self.tau_act[ii0] = 0
        self.xfrc[ii0] = 0
        self.sample_cmd(idx)
        self.update_goal_cmd()
        self.vint[ii0] = 0; self.last_a[ii0] = 0; self.has_prev[ii0] = False; self.steps[ii0] = 0
        self.tq[ii0] = 1 + (self.U(n, lo=-0.15, hi=0.15) if self.dr else 0.0)''')
rep('''        if self.dr:
            ii = idx.nonzero()[:, 0]
            self.t_mass[ii] = self.mass_tab''', '''        if self.dr:
            ii = ii0
            self.t_mass[ii] = self.mass_tab''')
rep('''        if self.surfaces:
            ii = idx.nonzero()[:, 0]
            sid =''', '''        if self.surfaces:
            ii = ii0
            sid =''')
rep('''        self.hist[idx] = 0
        self.push_left[idx] = 0
        self.push_t[idx] = self.U(n, lo=200, hi=400)''', '''        self.hist[ii0] = 0
        self.push_left[ii0] = 0
        self.push_t[ii0] = self.U(n, lo=200, hi=400)''')
rep('''            applied[late] = self.prev_a[late]''', '''            applied = torch.where(late[:, None], self.prev_a, applied)''')

a = s.index("        if self.dr:   # random base pushes, 5 control steps long")
b = s.index('        if getattr(self, "ext_push", None) is not None:')
new2 = '''        if self.dr:   # random base pushes, 5 control steps long (sync free: sampled for all worlds, applied where a push starts)
            N = self.N
            self.push_t = self.push_t - 1
            start = (self.push_t <= 0) & (self.push_left <= 0)
            ang = self.U(N, hi=2 * np.pi); f = self.U(N, lo=self.drc['push_force'][0], hi=self.drc['push_force'][1])
            self.push_f = torch.where(start[:, None], torch.stack([f * torch.cos(ang), f * torch.sin(ang), torch.zeros_like(f)], 1), self.push_f)
            self.push_left = torch.where(start, torch.full_like(self.push_left, 5), self.push_left)
            self.push_t = torch.where(start, self.U(N, lo=200, hi=400), self.push_t)
            on = self.push_left > 0
            self.xfrc[:, self.base, :3] = torch.where(on[:, None], self.push_f, torch.zeros_like(self.push_f))
            self.push_left = self.push_left - on.long()
            if self.pull:      # sustained pulls: 3 to 12 N for 0.5 to 2 s, every 4 to 10 s
                self.pull_t = self.pull_t - 1
                st = (self.pull_t <= 0) & (self.pull_left <= 0)
                ang = self.U(N, hi=2 * np.pi); f = self.U(N, lo=3.0, hi=12.0)
                self.pull_f = torch.where(st[:, None], torch.stack([f * torch.cos(ang), f * torch.sin(ang), torch.zeros_like(f)], 1), self.pull_f)
                self.pull_left = torch.where(st, torch.randint(50, 200, (N,), device=DEV, generator=self.g), self.pull_left)
                self.pull_t = torch.where(st, self.U(N, lo=400, hi=1000), self.pull_t)
                on2 = self.pull_left > 0
                self.xfrc[:, self.base, :3] += torch.where(on2[:, None], self.pull_f, torch.zeros_like(self.pull_f))
                self.pull_left = self.pull_left - on2.long()
'''
s = s[:a] + new2 + s[b:]
open(p, 'w').write(s)
print("patched")
