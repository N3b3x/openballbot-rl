"""One-off patch: fully mask-based (host-sync-free) reset for train_dev.py."""
p = 'train_dev.py'
s = open(p).read()
a = s.index("    def reset(self, idx):")
b = s.index("    @staticmethod\n    def rotvec")
new = '''    def reset(self, idx):
        """Reset the worlds in idx (bool mask over N). Everything is drawn for all N worlds and merged with where / masked_fill_, so there is no host sync.
        Tensors that share memory with Warp arrays are updated in place with copy_."""
        N, mk = self.N, idx
        m1, m2, m3 = mk[:, None], mk[:, None, None], mk[:, None, None, None]
        if self.K > 1:      # new design for these worlds
            dn = torch.randint(0, self.K, (N,), device=DEV, generator=self.g)
            self.didx = torch.where(mk, dn, self.didx)
            for nm, st in self.dstack.items():
                cur = self.dw[nm]
                msk = mk.view(-1, *([1] * (cur.dim() - 1)))
                cur.copy_(torch.where(msk, st[dn], cur))
            self.tau_w = self.tau_tab[self.didx]; self.f_w = self.f_tab[self.didx]
        yaw = self.U(N, lo=-np.pi, hi=np.pi)
        q = self.q0_tab[self.didx].clone()
        q[:, 3] = torch.cos(yaw / 2); q[:, 4] = 0; q[:, 5] = 0; q[:, 6] = torch.sin(yaw / 2)
        if self.tiles:      # random tile, random start inside the flat start zone, z lifted by the local map height
            tid = torch.randint(0, self.tile_xy.shape[0], (N,), device=DEV, generator=self.g)
            self.cur_tile = torch.where(mk, tid, self.cur_tile)
            xs = self.tile_xy[tid, 0] + self.U(N, lo=-0.3, hi=0.3); ys = self.tile_xy[tid, 1] + self.U(N, lo=-0.3, hi=0.3)
            hz = self.hgt(xs, ys)
            for j in (0, 10):
                q[:, j] += xs; q[:, j + 1] += ys; q[:, j + 2] += hz
        elif self.terrain:      # random segment of the ridge, random x; z lifted by the local terrain height
            seg = torch.randint(0, len(START_Y), (N,), device=DEV, generator=self.g)
            sgn = torch.where(self.U(N) < 0.5, -1.0, 1.0)
            ys = torch.tensor(START_Y, device=DEV)[seg] * sgn + self.U(N, lo=-0.12, hi=0.12)
            if self.edge_p > 0:      # start shortly before one of the edges
                ed = torch.tensor(STEP_Y, device=DEV)[torch.randint(0, len(STEP_Y), (N,), device=DEV, generator=self.g)]
                ys = torch.where(self.U(N) < self.edge_p, sgn * (ed - self.U(N, lo=0.2, hi=0.45)), ys)
            xs = self.U(N, lo=-1.0, hi=1.0)
            hz = torch.zeros(N, device=DEV)
            for e, hh in zip(STEP_Y, STEP_H):
                hz = torch.where(ys.abs() >= e, torch.full_like(hz, hh), hz)
            for j in (0, 10):     # base and ball free joints (qpos 0:7, 10:17)
                q[:, j] += xs; q[:, j + 1] += ys; q[:, j + 2] += hz
        self.qpos.copy_(torch.where(m1, q, self.qpos))
        self.qvel.masked_fill_(m1, 0.0)
        self.warm.masked_fill_(m1, 0.0)
        self.ctrl.masked_fill_(m1, 0.0)
        if self.motor:
            self.tau_act.masked_fill_(m1, 0.0)
        self.xfrc.masked_fill_(m2, 0.0)
        self.sample_cmd(mk)
        self.update_goal_cmd()
        self.vint = torch.where(m1, torch.zeros_like(self.vint), self.vint)
        self.last_a = torch.where(m1, torch.zeros_like(self.last_a), self.last_a)
        self.has_prev = self.has_prev & ~mk
        self.steps = torch.where(mk, torch.zeros_like(self.steps), self.steps)
        self.tq = torch.where(mk, 1 + (self.U(N, lo=-0.15, hi=0.15) if self.dr else torch.zeros(N, device=DEV)), self.tq)
        if self.dr:
            new = self.mass_tab[self.didx].clone()
            for b in self.body_ids:
                new[:, b] = new[:, b] * (1 + self.U(N, lo=-0.15, hi=0.15))
            self.t_mass.copy_(torch.where(m1, new, self.t_mass))
            nf = self.nom_fric.unsqueeze(0).repeat(N, 1, 1)
            nf[:, :, 1] = self.nom_fric[:, 1][None] * (1 + self.U(N, lo=-0.2, hi=0.2))[:, None]
            self.t_fric.copy_(torch.where(m2, nf, self.t_fric))
            nd = self.damp_tab[self.didx].clone()
            lo, hi = self.drc["damping"]
            for k in range(3):
                nd[:, 6 + k] = self.U(N, lo=lo, hi=hi) * self.damp_tab[self.didx, 6 + k] / 0.1
            self.t_damp.copy_(torch.where(m1, nd, self.t_damp))
        if self.surfaces:
            sid = torch.randint(0, len(SURFACES), (N,), device=DEV, generator=self.g)
            self.surf_id = torch.where(mk, sid, self.surf_id)
            mu = self.surf_tab[sid, 0] * (1 + self.U(N, lo=-0.15, hi=0.15)); tc = self.surf_tab[sid, 1]
            for g in self.surf_geoms:
                self.t_gfric[:, g, 0] = torch.where(mk, mu, self.t_gfric[:, g, 0])
                self.t_gsolref[:, g, 0] = torch.where(mk, tc, self.t_gsolref[:, g, 0])
        self.hist = torch.where(m2, torch.zeros_like(self.hist), self.hist)
        self.push_left = torch.where(mk, torch.zeros_like(self.push_left), self.push_left)
        self.push_t = torch.where(mk, self.U(N, lo=200, hi=400), self.push_t)

'''
s = s[:a] + new + s[b:]
open(p, 'w').write(s)
print("patched reset")
