## What and why

<!-- One or two sentences. Link the issue if there is one. -->

## Path touched

- [ ] Current pipeline (`experiments/warp_ppo/`, `experiments/mjx_port/`, `scripts/study/`, `specs/`)
- [ ] ROS 2 package (`ros2/`)
- [ ] Documentation or website (`docs/`, `site/`)
- [ ] Legacy SB3 path (`ballbot_gym/`, `ballbot_rl/`, `configs/`, `tests/`)

## Checks

- [ ] Every number I added or changed is in `docs/research/tracking_study_log.md` with its S-number, and the text says it is a simulation result
- [ ] No claim implies a physical robot or hardware results
- [ ] ROS 2 changes: `cd ros2/ballbot_control && python -m pytest -q` passes
- [ ] Docs changes: `python scripts/docs/check_docs.py` passes
- [ ] Site changes: `python scripts/site/check_links.py` passes
- [ ] Large files (checkpoints, rollouts, recordings) are not added; they are git-ignored

## Notes for the reviewer

<!-- What to look at first, what was not tested. -->
