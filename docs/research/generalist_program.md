# Generalist ballbot controller: research program (simulation only)

Goal: one policy that tracks commanded velocity and paths, chases targets, resists pushes and pulls, adapts fast, and survives steps, ramps, drops and rough ground, across randomised robot designs. Pure PPO and pure LQR stay as baselines. No hardware claims.

## Infrastructure status (2026-11-05)
- GPU PPO on MuJoCo-Warp: 4096 worlds, ~100k to 140k env-steps/s, per-world domain randomisation (mass, friction, damping, torque, noise, latency, pushes), realistic motor model. CPU engine remains the reference; every GPU result gets a CPU cross-check.
- Terrain: one shared height-field in Warp, so per-world variety comes from a tiled map (`experiments/warp_ppo/tiles.py`): 8x8 tiles of 3 m, types flat, step up/down, small/large drop, ramp up/down, stairs, bump, rough, severity sampled per tile; a world picks a tile at reset and is truncated when it leaves the tile.
- Sensors on GPU: privileged height-scan teacher, VL53L9-style depth rays (BVH `mjw.rays`, 0.69 ms per 64 rays x 4096 worlds), cone and head mounts.
- Findings so far: blind PPO trained on a ridge crosses 2 cm steps at 0.6 m/s but not at 0.3 m/s (approach speed dominates); no perception gain yet under a speed-tracking reward (S26, S27).

## Workstreams
1. **Terrain curriculum and generalist training.** Tiled map, severity curriculum, per-type success tables (fall, crossed). Resample the map between iterations (needs a test that in-place height-field edits reach the collision BVH). Add Perlin tiles at several scales, step-down chains, and moving/pushed-floor cases later.
2. **Task interface.** Today: velocity command. Add (a) goal chase (target position, speed free), (b) path following (lookahead waypoints), (c) pulls and pushes (force profiles, sustained pulls). Speed must be free for terrain: reward progress, not speed error.
3. **Design randomisation per world.** Batch model fields across K compiled designs (scale, ball and base mass, COM height, torque, rotor inertia, wheel damping, grip) and sample a design per world; feed a normalised design vector or let an adaptation module infer it (RMA-style).
4. **Architecture search.** Compare MLP, MLP with history stack, recurrent (GRU), teacher-student with privileged terrain and design latents, depth encoder joint versus frozen autoencoder (Salehi), residual-on-LQR. Same budget, several seeds, one metric table.
5. **Perception.** Only after a goal-directed reward exists: blind vs teacher vs depth, mount placement (head, cones as modelled, cones re-aimed), resolution 8x8 / 18x14 / 54x42, noise and dropout, latency.
6. **Sim-to-real gap study (simulation-side).** Sensor model realism (IMU bias, encoder quantisation, delay), actuator model (lag, torque-speed, friction, backlash), contact model (anisotropic friction vs roller ring, PhysX cross-check), timing jitter, push/pull forces. Report sensitivity of the best policy to each gap factor and train with the factors randomised; quantify the CPU vs Warp vs PhysX spread as a floor on what sim agreement means.
7. **Design-space map.** Success and speed ratio versus design parameters for the generalist policy and for LQR, using the same per-type terrain table.

## Metrics (fixed before runs)
Per terrain type and severity: fall rate, crossed rate, velocity-tracking ratio at 0.3 / 0.6 / 1.0 m/s, push recovery at 0.5 / 1 / 1.5 m g, action smoothness (mean |delta a|), ball offset. At least 3 seeds for any claim in the paper; CPU cross-check on the final policies. Every table notes engine, DR on or off, noise on or off.

## Order of work
Phase 1: tiled generalist baseline (blind), per-type table, CPU cross-check. Phase 2: goal-directed reward and chase/path/pull tasks. Phase 3: per-world design randomisation. Phase 4: architecture comparison with teacher-student and history/recurrent. Phase 5: perception arms. Phase 6: sim-to-real sensitivity sweep and final robust policy; update paper and site.
