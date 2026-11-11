# Site and paper cross-check

Scope: `site/generalist.html`, `site/method.html`, the generated figures and tables, and `paper_v2/sec_generalist.tex` with its generated tables and numbers. Generated 2026-11-07 against the study log up to S63 and the files in `outputs/study/warp_ppo/`.

## S69 correction applied (read first)

Per log entry S69 (stale `body_subtreemass` under mass randomisation), every result obtained with randomisation on before the fix is superseded. Applied to the site and paper:
- **Removed from the pages and paper, replaced by PENDING:** the goal-task and tile-velocity tables (old Tables 2 and 3), the blind tile baseline, the recipe-ceiling, seed-noise and load-curve analyses and figures, the perception comparisons on the ridge and tiles (ridge table, goal bars), the per-design and per-kind training fall breakdown, the terrain-goal figure, the training-curve figure of the old runs, the wide-randomisation, soft-LQR and ablation text, and the "mass is the driver" reading. Their files were deleted from `site/figures/study/` and `paper_v2/`.
- **Kept:** the CPU baseline study (Table 0), the planar model, LQR and PPO derivations, the Setup page, the engine parity table (with a caveat: the policies were trained with randomisation before the fix, the table compares engines with randomisation off), the encoder pretraining table (caveat: frames collected on the affected engine), the clean flat benchmark (Table 1) and the flat-only authority figure (caveat: the learned policies were trained before the fix), the tile geometry, the LQR calibration, the videos (caveat: render settings unrecorded, policies trained before the fix).
- **Added:** the engine-pitfall paragraph on the Setup page (section 1) and in the paper (section on engines), and three tables (goal, tile velocity, robustness) plus a training-progress figure that read `outputs/study/warp_ppo/g12_*/battery.json` and show PENDING until it exists. `site/build_data.py` no longer builds any randomisation-on table from pre-fix files; the snapshot copies in `site/data/eval_snapshot/` of those files are unused.
- **Automatic section below:** the checks for the removed tables are gone (157 checks remain: flat benchmark, CPU table, parity, encoder, model, source and configuration statements).
- Items F1 to F6 and the thin-data list below were written before S69; the parts about randomised goal and tile results (the g7_lqrres_s0 goal files, `g2_s0` and `gfix_*` files, perception, seeds, g9 and the old numbering of the tables) are historical and only the engine-independent points remain relevant: F1 (the evaluation names do not record `PUSH_SCALE`, `DR_SCALE`, `TILES_FLAT`, `EVAL_NOEXIT`; `eval_battery.py` now stores them in `env_settings` and in file names), F2 (64 worlds on the flat row), the F3 entries on the flat benchmark, the model and the LQR ratios, F4 (log-only values), and thin-data items 1 to 7 and 9 (second seeds, held-out tiles and designs, CPU re-evaluation, LQR alone on the tile protocols, S56 as a data file, clip metadata).

## S104 applied: navigation on the recommended checkpoint (2026-11-09)

New Table 37 in chapter 10 (`#nav26`) from `g26_ret_rs025_mu01_s1/navS_{blind,oracle,tof18}.json` and `navS_{oracle,tof18}_pp.json` (the generator asserts run, 528 worlds, 2000 steps, smooth planner, push scale, aimed 18 x 14 sensor): reached / fell / stuck / onto a block, median time and heading-rate RMS; all 24 S104 values match. Tables 19 to 23 keep the g18 measurement and their captions now say "earlier measurement: controller g18_ret_rs025_s1 trained and evaluated at friction 0.001". The TL;DR, the "At a glance" navigation tile, conclusions bullet 7 and the paper now quote the recommended checkpoint (ToF 0.84, perfect map 0.89, blind 0.09; about half under pushes and pulls). The S100 PENDING line for navigation is gone.

## S103 final applied: the speed layer is worth 0.03 to 0.05 (2026-11-09)

Table 36 is now the 40 s table on g26 seed 1 at its own friction (`g26_ret_rs025_mu01_s1/gov_vel03_ps0_{none,oracle_v005,tof18_v005}_40s.json`, each asserting 4000 steps and `pair_mu1` 0.01): no layer / perfect map / ToF falls 0.38 / 0.33 / 0.35, stairs 0.56 / 0.43 / 0.47, tile exits 0.62 / 0.58 / 0.54, progress 1.35 / 1.31 / 1.28 m; all 24 cells match the S103 addenda. The goal-task 30 s values (`gov_goal_ps0_{none,tof18,tof18_v01}_30s.json`: 0.24 / 0.76, 0.23 / 0.62, 0.25 / 0.74) match and are quoted in prose. The 15 s rows (g18 three seeds and the S102 matched pair) are no longer tabulated; their values are still computed and checked, and the S102 family equality is cited as a 15 s result. The "under re-test" lines are replaced by the final reading everywhere (Table 36 "what this means", traction sentence, chapter 12, conclusions bullet 6, paper paragraph, traction sentence and abstract): the 15 s halving was an exposure artefact; the real effect is 0.03 to 0.05 (differences of the rates as printed, as in the log), mostly on stairs and bumps (the log's "everything else within 0.06" holds for step ups, drops and flat tiles; bumps move 0.07 with the perfect map), at a cost of 0.04 to 0.08 in tile exits; obstacle falls are set by obstacle size against ball size; routing and a larger ball are the tools that work. The ToF estimator precision 0.73 and recall 0.85 are quoted from S103 with a dagger (no site data file). The S100 PENDING line now names only the navigation re-run on the recommended checkpoint; chapter 10 is unchanged.

## S103 addendum applied: speed-layer claims suspended (2026-11-09)

Every statement that the speed layer or governor halves obstacle falls is replaced by "Under re-test: the speed-layer result was measured with 15 s episodes, which may not give a slowed robot time to reach its feature (S103 addendum); a 40 s re-test is running.": the Table 36 "what this means" (the matched-pair family comparison is kept, same protocol), the traction subsection's closing sentence, the chapter 12 sentence, conclusions bullet 6, the paper's S100 paragraph, traction sentence and abstract sentence. Table 36's caption says "under re-test"; its numbers stay (they are what the files hold). The S100 PENDING line now leads with the 40 s re-test. Table 35 (speed against falls, 20 s, every speed reaches its feature) and its "what this means" stay; neither draws a governor conclusion. results.html, architecture.html and method.html have no halving claim (only the LQR reference governor). index.html is the coordinator's.

## S102 applied: the S101 family gap was an evaluation bug (2026-11-09)

eval_governor.py did not apply a run's own `pair_mu1`, so every governor and step-size result on a 0.01-trained run (g26, g27, g28) ran at 0.001. Applied:
- **Removed:** the two-family decision box, the g26 rows of Table 36 (`gov_vel03_ps0_*.json`, no longer read by the generator), the S101 addendum test (`gov_mu0.001_vel03_ps0_*.json`), and every family-gap, "0.14 to 0.17", accidental-penalty / slow-approach-curriculum and policy-against-model statement on generalist.html and in the paper (none existed on method.html; index.html cites are the coordinator's). No g27 result appears anywhere; chapter 9 keeps only its PENDING line.
- **Restored:** a single recommendation box in chapter 8 ("Decision: the retrained seeds are the recommended checkpoints"): g26 s0, s1, s2, with the battery ranges, standing, and equality on the obstacle course with a speed layer, plus the sentence that the S101 gap was an evaluation bug, fixed (S102). Conclusions bullet 4 and the chapter 8 callout follow; the paper decision sentence is restored.
- **Table 36 now:** the three g18 rows (`gov_mu01_vel03_ps0_{none,oracle_v005}.json`, explicit 0.01; all 42 cells match S101's g18 rows, which S102 keeps) plus the S102 matched pair from `behav_mu01_{none,oracle}.json` (both record `pair_mu1` 0.01; g18 s1 0.35 / 0.17, g26 s1 0.37 / 0.16, matching S102) and two PENDING rows for g26 seeds 0 and 2 (q49 re-run). The S100 PENDING line names that re-run, the sensor-driven speed layer and the "refuse" mode.

## S101 applied [superseded by S102]: the speed layer on six seeds; two checkpoint families, no single winner (2026-11-09)

- Table 36 is now the six-seed table: g18 seeds 0, 1, 2 (trained at 0.001, evaluated at 0.01; `gov_mu01_vel03_ps0_{none,oracle_v005}.json`) and g26 seeds 0, 1, 2 (trained at 0.01; `gov_vel03_ps0_{none,oracle_v005}.json`), each row without and with the governor (falls all tiles / step up / bump / stairs / large drop / flat / progress in 15 s), protocol in the caption. All 78 cells the S101 table gives (its "n.r." cells are read from the files and not checked) match; progress losses 0.22 (g18) and 0.26 (g26) match. The single-seed 0.1 m/s governor value (0.26, g18 s1) stays as prose.
- The chapter 8 decision box is rewritten ("Decision: two checkpoint families, no single winner"): g26 is the clean re-baseline on the battery and the only family with a reproducible standing number; on the 15 s obstacle course with a speed layer the g18 checkpoints evaluated at 0.01 are better by 0.14 to 0.17 (0.18 to 0.19 against 0.32 to 0.35) and by 0.01 to 0.05 without it, on all three seeds each; the 8 s battery and standing show no gap; cause unknown; deployment choice by use (standing and goals: either; feature-rich terrain with a speed layer: g18 at 0.01). The gaps are computed as range differences between the families (g26 min minus g18 max, g26 max minus g18 min). The chapter 8 callout's "Checkpoints" sentence, conclusions bullets 4 and 6, the Table 36 lead and "what this means" (gain on every seed; the family gap; candidates for the cause; policy-against-model test pending) and the S100 PENDING line (g26 governor done; sensor-driven governor, "refuse" mode and the policy-against-model test pending) follow. The paper's correction paragraph and checkpoint decision are mirrored.
- S101 addendum: the policy-against-model test (`g26_ret_rs025_mu01_s0/gov_mu0.001_vel03_ps0_{none,oracle_v005}.json`, 0.40 and 0.32, matching the log) is quoted in the Table 36 "what this means", the decision box, conclusions bullet 4 and the paper: the gap follows the policy, not the evaluation model; the accidental-penalty explanation is stated as a hypothesis; the pending line now names a slow-approach curriculum instead of the test. index.html cites were done by the coordinator.

## S100 applied: speed does matter on the corrected contact model (2026-11-09)

- New subsection in chapter 8 (`#speed`, "Correction: speed does matter on the corrected contact model") between the size split (Table 15) and the traction subsection: Table 35 (falls against commanded speed at friction 0.01, `g18_ret_rs025_s1/speed_mu01_{0.1,0.2,0.3,0.6,0.9}.json`, 20 s, 176 worlds per kind, with the superseded Table 14 values at 0.001 in a last column) and Table 36 (the speed layer on the privileged height map, `gov_mu01_vel03_ps0_{none,oracle,oracle_v005}.json`, 15 s, 176 worlds per kind, with falls, per-kind falls, mean speed scale, progress in 15 s and the superseded S87 values). All 30 cells of the S100 table and all 24 of the addendum table match the files; the progress loss (0.21) is computed from the two progress fields. A PENDING line names the sensor-driven governor, the governor on the g26 checkpoints and the "refuse" mode. The paper has the same correction paragraph with `tab_gen_speed_mu01.tex` and `tab_gen_gov_mu01.tex`.
- Corrected claims: the chapter 8 lead and "what this means" after Table 14 now say the no-speed-dependence and governor-does-not-help readings were made on the old contact model, were partly the standing blow-up, and are superseded by Tables 35 and 36; the size dependence (Table 15, S88 re-check within 0.11) and the S86 exposure lesson stand. The traction subsection is retitled "A traction limit, and what foresight can and cannot do" (id `#traction` unchanged) and its closing sentence and the chapter 12 "no amount of looking ahead" sentence now say that above the limit foresight changes nothing and below it slowing before a feature does help. Conclusions bullet 6 gained the S100 numbers. Paper: the retainer-section sentence, the traction paragraph's closing sentence and the abstract sentence "depend on obstacle size, not approach speed" were rewritten the same way.
- index.html line 142 ("Obstacle falls depend on obstacle size, not approach speed ... S87") was corrected as asked, with the S100 numbers and cite "S87, S100"; the other index.html and results.html lines ("not climbed at any speed", "fail at any approach speed") describe the size limit and stand. architecture.html has no speed claim; its "expose every world to the feature" lesson stands. method.html's reward note ("nothing in the reward values looking ahead") stands.

## S99 close-out applied: g26 checkpoints, Table 32 at 0.01 (2026-11-09)

- Table 34 has all three retrained seeds (`g26_ret_rs025_mu01_s2` added from its `eval2_vel_0.0|0.3_dr_surf_ps0_flat_noexit.json`: standing 0.025, kb025 0.24, bal4 0.04; driving 0.064, kb025 0.49, bal4 0.21; non-finite 0.000; all match the close-out). The PENDING g26 block is replaced by a decision box: g26 s0, s1, s2 replace g18 s0, s1, s2 as the recommended checkpoints, same recipe, corrected contact model; g18 numbers stay valid 8 s and driving comparisons, their standing numbers are void. The three g26 runs joined `RUNSPEC` (family `ret025mu`), so Tables 1 and 2 carry them as rows "corrected contact model", and their battery ranges (from `g26_ret_rs025_mu01_s{0,1,2}/battery.json`) are quoted in the box and the paper; all 30 range checks against the close-out match (goal, velocity falls at three randomisation scales, flat achieved per speed, standing and driving).
- Headline numbers moved to the g26 ranges on generalist.html (each place says which): the two "At a glance" stat tiles (tracking 0.85 to 1.00 and goal 0.74 / 0.45, with the g18 values at 0.001 given alongside), the chapter 8 recommended-controller callout (new "Checkpoints" sentence: goal 0.73 to 0.74 / 0.45 to 0.46, flat 0.85 to 1.00, error at most 0.054 m/s) and conclusions bullet 4. Kept at g18 (0.001) with an explicit cite: the authority curve and the recommended-controller paragraph numbers inside chapter 8 (recipe comparison at 0.001), the depth-sensor comparison in chapter 12 (g20 at 0.001 against g18 at 0.001, the matched pair), the chapter 5 flat-fall sentence, and the held-out design, navigation and smooth-follower results (g18 s1 frozen at 0.001, not re-run).
- Table 32 is rebuilt from `g24_scale_rs025_s0/battery_mu0.01.json` and `_s1/battery_mu0.01.json` (goal 0.74 / 0.47 and 0.70 / 0.44; flat achieved 0.91 to 0.99; no log row, checked against the files as read) next to the three g26 checkpoints at 0.01; its PENDING line is gone and chapter 9 now notes `g27_scale_rs025_mu01_s0` (training, progress read from its log) and the held-out 0.5 scale at 0.01 as pending.
- Not mine: index.html key numbers and results.html line 45 still quote the g18 ranges (tracking 0.99 to 1.01, goal 0.71 to 0.76 / 0.42 to 0.44, S83 / S84); recommended replacements are in the report to the coordinator.

## S99 applied: ball-size study at friction 0.01; retrained seeds in Table 34 (2026-11-09)

- Chapter 9 (site and paper) is regenerated from `g24_scale_rs025_s0` and `_s1` `stepsize_mu01_{s0.75,nom,s1.5,s2.0}.json` (policies trained at 0.001, evaluated at wheel-ball free-rolling friction 0.01; these files carry no `pair_mu1` in env_settings, so the friction value is taken from the file name and the S99 addendum). Table 30, Table 31 and Figure 10 (light, dark, paper) captions say so; the held-out 0.5 scale has no 0.01 file and stays in Table 30 from `stepsize_s0.5.json` with the label "at friction 0.001 only". All 24 cells of the S99 addendum (overall, large-half step up, large-half bump; seed 0 | seed 1; four sizes) match the files. A sentence states the change from the first pass, computed from the 0.001 files of the same seeds: small robots within 0.03 overall, the large robots lose 0.05 to 0.06 overall and up to 0.15 (largest per-seed change) at the large steps; ordering unchanged.
- Re-derived tokens: pooled 50 percent point 0.17 to 0.29 r (was 0.15 to 0.26), mean 0.24 r (was 0.21); rule of thumb `sc.rule_lo` 3.4 (unchanged), `sc.rule_hi` 4.2 (was 4.8). The bounds.html callout that cites these values needs the new `sc.rule_hi`; it is the other writer's page. The check "50 percent point" against the log's 0.18 to 0.22 r (seed 0, friction 0.001) now differs on the maximum and the mean, as recorded.
- Table 32 (scale battery) stays at friction 0.001 and its caption, lead-in and a PENDING line say so; it switches to the 0.01 battery when `battery_mu0.01.json` exists for both scale seeds.
- Table 34 gained the retrained seeds `g26_ret_rs025_mu01_s0` and `_s1` (`eval2_vel_0.0|0.3_dr_surf_ps0_flat_noexit.json`, args.json `pair_mu1` 0.01) and a non-finite column (0.000 where recorded, "not recorded" for the g18 rows); S99 values match (standing 0.028 / 0.030, kb025 0.27 / 0.28, bal4 0.04 / 0.06; driving 0.062, kb025 0.48, bal4 0.20; non-finite 0.000). The PENDING block names `g26_ret_rs025_mu01_s2` (training) and quotes the seed 0 battery from `g26_ret_rs025_mu01_s0/battery.json` (goal 0.74 / 0.45, 0.72 / 0.41, 0.68 / 0.33; flat achieved 0.90 to 0.99; all match the S99 addendum). Chapter 8 "what this means" and the paper correction paragraph say the retrained recipe reproduces the two weak designs.

## S98 applied: every 20 s standing number withdrawn (2026-11-09)

Per log entry S98 the 20 s standing fall rates of S93 (LQR gain sweep, LQR-alone standing), S94, S96 and S97 were float32 blow-ups of the 1000 to 1 anisotropic wheel-ball friction (0.001 along the roller axis, 1.0 across), counted as falls by the termination rule; driving at 0.3 m/s is barely affected and the 8 s battery not at all. Applied:
- **Removed from the pages, paper and generator:** Table 25 (base rate by design, standing and driving), Table 26 (LQR gain sweep), Table 33 (standing across seeds), the PENDING g25 block, their lead-ins and "what this means" lines, the value keys `base.*`, `lqrs.*`, `gs.*`, `st.*`, `ep20.*`, the paper paragraph "The long-horizon base rate" and `tab_gen_baserate.tex`, `tab_gen_gainsweep.tex`, `tab_gen_standing.tex` (deleted). The chapter 5 methods-note sentence and the conclusions bullet that quoted them were rewritten; the "How to read" box now names two corrections; the limitations box names the friction change. The old table numbers 25, 26 and 33 no longer appear on generalist.html (method.html does not reference them either).
- **Added:** a correction box in chapter 8 (`#standing`: what was measured, why it was wrong, what was ruled out, the fix) with daggered diagnostic values from S98 (91 percent of the worst seed's standing falls neither tip nor unseating; non-finite fraction 0.127 at 0.001 and 0 of 880 at 0.01), new Table 34 from `g18_ret_rs025_s1` and `_s2` `eval2_vel_0.0_dr_surf_ps0_flat_noexit_mu0.01.json` and `eval2_vel_0.3_dr_surf_ps0_flat_noexit_mu0.01.json` (env_settings `pair_mu1` 0.01, `tiles_flat`, `no_exit`, 2000 steps, 32 worlds per design asserted by the generator), a PENDING block for `g26_ret_rs025_mu01_s0/s1` (progress read from their logs), the paper paragraph "Correction: the 20 s standing falls were a simulator blow-up" with `tab_gen_standing_mu01.tex`, a second engine-pitfall section on method.html (section 1), the method.html Table E row (friction 0.01 since S98, args.json rule, non-finite states classified separately) and the rewritten method.html design "what this means" (kb025 unseats, bal4 falls in about a fifth of driving worlds, kb4 and bal025 stand at 0.00). The method.html integral sentence now says the "essential for standing" claim is unverified.
- **Checks:** all 18 cells of the S98 addendum table match the files (standing all 0.027 / 0.039; kb4 and bal025 0.00; kb025 0.24 / 0.29; bal4 0.05 / 0.13; driving all 0.063 / 0.075; kb025 0.50 / 0.54; bal4 0.18 / 0.23; nominal 0.00 / 0.02). Seed 0 of the recommended controller has no 0.01 files yet, so it is not in Table 34.
- **Not verified / left to others:** whether the evaluation files classify non-finite states separately (stated from the coordinator's note; the mu0.01 files carry no separate field that the generator reads); the "essential velocity integral" claim is unverified rather than disproved; bounds.html (lines 38, 167 to 194: LQR envelope and the S93 LQR-alone standing table and integral rule), glossary.html `#integral` and lessons.html belong to the other writer; index.html line 137 says "the light-ball design kb025 falls on flat ground and LQR gains cannot fix it (S92, S93)", of which the second clause is void under S98 (designer's page); results.html lines 78 and 87 quote S87 and the navigation study, no standing numbers.

## S94 and S95 applied (long-horizon standing across seeds; ball size against step size)

S94: new Table 33 in chapter 8 (`eval2_vel_0.0_dr_surf_ps0_flat_noexit.json` and `eval2_vel_0.3_dr_surf_ps0_flat_noexit.json` of g18_ret_rs025_s0, g18_ret_rs025_s1 and g24_ret_rs025_ep20_s0; `g24_ret_rs025_ep20_s0/battery.json` for its standard battery). Every sentence that said the nominal robot falls 0.00 standing now says that this holds for seed 1 only (seed 0: 0.15) and quotes the seed range; the Table 25 caption names the seed and points to Table 33; the conclusions bullet and the paper paragraph say that standing is seed dependent, driving is reproducible, and 20 s episodes alone are not a fix. `g25_ret_rs025_stand_s0` is PENDING in the chapter 8 block.

S95: new chapter 9 "Ball size against step size" (site) and subsection `sec:scale` (paper), between the retainer chapter and navigation; navigation, designs, perception and conclusions are renumbered 10 to 13 (section ids unchanged), Figures 10 and 11 of those chapters became 11 and 12. Sources: `g24_scale_rs025_s0/stepsize_{s0.5,s0.75,nom,s1.5,s2.0}.json` (`by_severity` halves for Table 30, `per_world` heights binned at 1 cm for Table 31 and at 0.5 cm for Figure 10, redrawn in the site's style with light, dark and paper variants; dotted lines at 0.29 r), `g24_scale_rs025_s0/battery.json` for Table 32, and clips `c7_step4cm_s075` / `c7_step4cm_s2` (sidecar JSON from `outputs/study/warp_ppo/videos/`; the video card now prints the clip's design). The 50 percent point of each curve is found by linear interpolation of the 0.5 cm-bin curve per size and feature: 0.15 to 0.25 r (mean 0.21 r) against the log's "roughly 0.18 to 0.22 r"; the pages quote the computed range and mean, and the design rule (ball radius 3.4 to 4.8 times the tallest step) is 1/0.29 and 1/(mean 50 percent point). Known difference: the log's first-pass S95 table was computed from an earlier draw of the evaluation; the `stepsize_*.json` files were regenerated with per-world heights (S95 addendum) and differ by up to 0.17 on single cells (s2.0 stairs, step up), while the 1 cm-bin addendum and the 18 cm-ball figure values match the files exactly. The files are used. Seed 1 of the scale policy (`g24_scale_rs025_s1/stepsize_{s0.75,nom,s1.5,s2.0}.json` and `battery.json`) was incorporated the same day: Table 30 cells are the mean of the two seeds with a last column giving each seed's all-tiles fall (0.5 scale: seed 0 only, no seed 1 file), Table 31 and Figure 10 pool the worlds of both seeds (n per bin 36 to 128) and the figure draws a vertical bar between the two seeds' values at every bin, Table 32 has one row per seed (seed 1: goal 0.69 / 0.43, flat achieved 0.86 to 0.99), and the 50 percent point is recomputed on the pooled curves (0.16 to 0.26 r, mean 0.21 r; rule 3.4 to 4.8). The coordinator's seed-1 numbers (overall by design 0.48 / 0.35 / 0.26 / 0.18; large-step 1.00 / 0.98 / 0.73 / 0.65; large-bump 1.00 / 0.98 / 0.53 / 0.45) all match the files (checks "S95 seed 1"). The seed-0 checks against the addendum bins and the 18 cm-ball figure values are kept on seed 0 alone. The PENDING marker for seed 1 is gone; the bounds.html cite line now says "two training seeds of one recipe". A "Design for your ballbot" callout for the rules page was not added: that page belongs to the writer (the chapter 9 callout holds the text).

## S72 to S75 applied (corrected-engine results)

The PENDING chapters were filled from `outputs/study/warp_ppo/<run>/battery.json` of g12_res_s0/s1, g12_dr175_s0/s1, g12_soft175_s0/s1, g13_softrs075_s0 and g12_pure_s0. Tables 1 to 4, Figures 1 to 5, the prose numbers and the paper section are generated from those files; the old pre-fix flat rows (g2 to g9 policies), the engine policy-transfer table and the old clips (`s46_*`, `s61_*`) were removed; the new clips are `c1_goal_ramp`, `c1_goal_step`, `c1_goal_flat`. Checks against S72 to S74 (about 100 values) all match; the only rounding notes are the median times, which the log prints to one decimal. Still PENDING: the g14 variants (training, no battery.json yet), perception arms on the corrected engine, held-out tile seeds and designs. Because `battery.json` stores `env_settings`, the file-name collision problem of F1 no longer applies to these runs. The old per-item notes below (F1 to F6, thin data) were written before S69 and refer to superseded files.

## Flat re-run applied (S92, all 34 batteries)

`python site/build_data.py --refresh` removed the freeze file; Table 1, Figures 1, 2 and 4, Tables 10 and 11 and every sentence quoting a flat fall rate now use the re-run rows (interior tiles). The recommended controller falls 0.00 at every speed on all three seeds; the highest learned flat fall rate in the re-run is 0.11 (g12_pure_s1 at 0.3 m/s), and most rows are 0.00 to 0.02. The re-run is a fresh sample: its steady-state error and achieved speed moved against the earlier files by 0.001 to 0.01 m/s and up to 0.02 in ratio (for example the recommended controller's error at 0.9 m/s is now 0.036 and 0.034 against the logged 0.043 and 0.039), so the automatic checks no longer compare flat rows with S59 to S91 and a single supersession note stands at the top of the automatic section. The LQR-alone row was then re-measured on interior tiles (S92 addendum: 0.3 m/s falls 0.00, error 0.014, achieved 1.00; 0.6 m/s falls 0.00, error 0.017, achieved 1.00; 0.9 m/s falls 0.89, a real speed wall); the printed rows were then saved to `outputs/study/warp_ppo/lqr_flat_interior.log`, which the build parses (0.9 m/s: falls 0.91, error 0.026 and achieved 1.00 over the surviving worlds; the addendum text says 0.89), so Table 1, Figure 1 and the paper take the file values without a dagger, and the S56 drop-off caveat is removed. The LQR-against-pure-learning accuracy ratio on the pages follows from the new row (9 to 15 times at 0.3 and 0.6 m/s). Not re-run, shown with a caveat: the zero-shot arrestor flat row of g12_dr175_s0 (`eval2_vel_*_ret0-0-0-100_flat_noexit.json`). The battery flat rows do not carry a falls_at_map_edge field (the underlying eval2 files do), so the caption does not claim one.

## Restructure for the site uplift, with S92 and S93 (2026-11-08)

`generalist.html` and `method.html` were rewritten for readability (TL;DR box, a plain-language lead-in before every table and figure, terms defined on first use with a `.term` link to `glossary.html#id`, one idea per section, a "what this means" line after every result). Every `<!-- BUILD:name -->` block and every value token was kept; the rebuild reports no missing keys. Two sections were added from the log: S92 as a methods note (chapter 5 of the results page and section 8 of the method page: border tiles made robots drive off the 24 m map; recorded flat falls were mostly map-edge drop-offs, `eval2_vel_0.3_dr_surf_ps0_drs1_flat_noexit.json` holds the demonstration with 74 percent at the edge; tracking metrics unaffected) and S93 as a subsection of chapter 8 (Table 25, the 20 s base rate by design from the interior-tile files with and without the residual; Table 26, the LQR gain sweep parsed from `outputs/study/warp_ppo/lqr_sweep.log`; decision to keep R 1 and integral gain 1, kb025 reported separately; the long-episode and scale runs PENDING). **Flat fall column frozen:** the flat rows of every battery are being re-run on interior tiles (`flat_rerun.log`); to keep Table 1 from mixing old and new rows, `build_data.py` reads the pre-correction flat values from `site/data/flat_frozen.json` (written from the 18:30 build) and labels the column; `python site/build_data.py --refresh` deletes the freeze file and switches to the live rows. The S92 check for zlow standing compares against the log's upper bound (at most 0.04) rather than a printed value.

## S90 and S91 applied (smooth following, planner-in-the-loop null result)

Added to chapter 9: Table 21 (smooth follower against the wavefront planner on the privileged map, `nav_oracle.json` and `navS_oracle.json`), Table 22 (ToF variants, `navS_tof18*.json`), Table 23 (planner-trained g23_nav_rs025_s0/s1 against the frozen controller, `navS_*.json` in each run folder), and the conclusions. The S89 wavefront file has no heading-rate field, so its 14.4 rad/s carries a dagger (log value). One rounding note: the S90 table prints the smooth follower on the privileged map as 0.85 / 0.14 / 0.01 / 0.02 (reached / fell / stuck / on block) while the file holds 0.858 / 0.138 / 0.004 / 0.040; the S91 table, which lists the same condition, prints 0.86 / 0.14 / 0.04 and matches the file, so the checks use the S91 values. The two S90 faults are one methods sentence; the rejected variants are one sentence with their numbers in Table 22. Later tables renumbered: designs 24, perception 25 to 27.

## S89 applied (terrain-aware navigation and sensors)

New chapter 9 "Terrain-aware navigation and sensors" (site) and subsection (paper): Table 19 (undisturbed, `g18_ret_rs025_s1/nav_*.json`), Table 20 (pushes and pulls and the wider-margin runs, `nav_*_pp*.json`, `nav_tof18_hw*.json`), Figure 10, and clips c6_nav_blind / oracle / tof18 (no sidecar JSON; their metadata is set in `CLIP_META` in build_data.py from the coordinator's description). `nav_oracle_wave.json` (written at 15:15, success 0.25, stuck 0.69) is an earlier run superseded by `nav_oracle.json` (15:18, the S89 value 0.91) and is not used. All S89 values checked match (611 checks, only the two model notes differ). Later chapters renumbered: designs 10 (Table 21, Figure 11), perception 11 (Tables 22 to 24), conclusions 12.

## S85-R and S88 applied

S85-R reinstates chapter 9 (unseen robot designs) from the re-written `eval2_goal_goal_dr_surf[_pull]_des-s1.5-tau3.json` files: Table 19 and Figure 10, with the earlier design-vector bug mentioned only in a methods note (site) and a footnote (paper); the implications callout is restored as simulation-only hypotheses (LQR + residual, arrestors, torque headroom, ball size and friction). S88 is added to "Where it still falls": the traction model with its derivation and Figure 9 (analytical; h_max = r(1 - cos 45 deg) = 2.6 / 4.0 cm), Table 16 (ball size, `stepsize_nom.json` and `stepsize_s1.5.json`), Table 17 (privileged height scan against blind counterparts, g22_* batteries and `eval2_vel_0.3_dr_surf_ps0.json` by kind), Table 18 (pure learning stalls rather than climbs, `stepsize_nom.json` of g16_pure_ret100_s1 and g18_ret_rs025_s1). All S85-R and S88 values checked match (561 checks, only the two model notes differ). Perception tables renumbered 20 to 22.

## S85 suspended, S86 withdrawn, S87 applied

S85 (unseen robot designs) is suspended in the log (design-vector bug in the held-out-design evaluation): chapter 9 now holds only a note that the evaluation is being re-run; Table 14 (designs) and Figure 9 were removed with their files, and the "pure learning collapses on unseen designs", "implications for a real robot" and motor-torque-headroom statements were removed from the summary, conclusions and abstract. The S83 held-out-design addendum shares the fault and was never on the site. S86 (falls depend on approach speed) is withdrawn and was never used on the site or in the paper. S87 is added to "Where it still falls" (site) and the paper: Table 14 (falls against commanded speed with 20 s episodes, `speed_long_{0.1,0.2,0.3}.json`), Table 15 (falls against feature size, `speed_long_0.3_sev.json`), and the governor result (`gov_vel03_ps0_none.json`, `gov_vel03_ps0_oracle.json`); all S87 values checked match. Perception tables renumbered 16 to 18. The S84 and S85 section below is historical for its S85 part.

## S84 and S85 applied (three seeds, 400M, depth on top, unseen designs)

Recommended controller now three seeds (g18_ret_rs025_s0/s1/s2) and the 400M run (g19_ret_rs025_400m_s0): PENDING markers replaced. Held-out terrain rows for the recommended controller added to Table 12. New Table 13 (per-obstacle falls, `eval2_vel_0.3_dr_surf_ps0.json` by kind, matching the S83 and S84 addenda). Depth on top (g20_ret_rs025_depth_s0) added to the perception chapter as Table 17. New chapter 9 "Unseen robot designs" with Table 14 and Figure 9 from `eval2_goal_goal_dr_surf[_pull]_des-s1.5-tau3.json` (by-design success; 160 worlds per design), and an "implications for a real robot" paragraph marked as simulation-only hypotheses. The g21 runs train on 12 designs, so their battery goal rows hold 1920 worlds (their labels say so) and tau3 is not unseen for them. Perception is chapter 10 (Tables 15 to 17), conclusions chapter 11. All S84 and S85 values checked match (476 checks, only the two known model notes differ).

## S81 to S83 and the S80 addendum applied (retainer trade-off and recommended controller)

Added to chapter 8 of the site and the retainer subsection of the paper: pure learning with the retainer (g16_pure_ret100_s0/s1), tracking weight x3 (g17_ret100_trk3_s0), the authority curve (g18_ret_rs015_s0, rs025_s0/s1, rs035_s0, g18_ret_soft_s0, g16_ret100_s0/s1) as Table 11 and Figure 8, the retainer on held-out terrain (Table 12, `eval2_goal_goal_dr_surf*_ts1.json`), the recommended-controller callout (authority 0.25 with the retainer; third seed and 400M run PENDING), and clips c3_step_up_pure_nocage / cage and c4_step_up_rs025_cage / c4_ramp_vel06_rs025_cage. Note: the two c3 clips are different policies (g12_pure_s0 trained without arrestors, g16_pure_ret100_s1 trained with them), and the captions say so. `g18_ret_rs015_s0/battery.json` did not exist yet at build time (the battery was still running); `build_data.py` assembles the same structure from the eval2 files already written, and the tile velocity at DR 2.5 for that run shows n/a until the file appears. All S81 to S83 values checked match. Perception tables renumbered 13 and 14.

## S79 and S80 applied (fall diagnosis and ball retainer)

New chapter 8 "Why it falls, and a ball retainer" (site) and subsection of the same name (paper): Tables 6 (fall diagnosis, `g12_dr175_s0/fallcause_ps0.json`, `fallcause_ps1.json`, `fallcause_ps1_lqronly.json`), 8 (mount sweep, `fallcause_ps*_ret0-0-0-<phi>.json`; the 135 deg row is `ret0-0-0-0`, the diametrically opposite mount), 9 (roller friction), 10 (training with the retainer: `g16_ret100_s0/s1`, zero-shot `eval2_*ret0-0-0-100*.json`, `g12_pure_s1`), Figures 6 (geometry) and 7, and the four `c2_*` clips. Floor clearance is computed from R + (R + r_a) cos(phi) - r_a (6.5 / 3.3 / 1.2 cm, matching S79); the floor-touch tilts (57 and 21 deg) are quoted from the log. The S79 saturation values in the log now match the files (0.7 % / 0.4 %), and S78 matches the current g15_edge_depth_s0/battery.json. The S79 settle bugs and the height-field artefact appear only as one footnote sentence. Perception moved to chapter 9 and its tables were renumbered 11 and 12.

## How the numbers get onto the pages

1. `python site/build_data.py` reads the result files (`eval2_*.json`, `log.csv`, `args.json`, `eval_*.txt`, `percep/enc_v1.json`, `wpdr_*/eval_{cpu,warp}.json`, `tracking_summary.csv`, the engine benchmark JSONs) and the code that defines the setup (`tiles.py`, `train_dev.py`, `configs/env/dz_*.yaml`, the planar model JSON).
2. It rewrites the tables (`<!-- BUILD:name -->` blocks), the figures (`site/figures/study/*.svg` light and dark, `paper_v2/figures/study_*.pdf`), the paper tables (`paper_v2/tab_gen_*.tex`), and every number quoted in prose (`{{key:decimals}}` tokens on the pages, `\gv{key}` in `paper_v2/gen_values.tex`). All values also go to `site/data/study.json`.
3. It compares each value with the number printed in the lab log and writes the result in the automatic section below. Differences are listed first.
4. Evaluation files are read from the frozen copy `site/data/eval_snapshot/` (see finding F1); `--refresh` re-copies the live files.

## Findings (read these first)

**F1. Evaluation JSON names do not encode the protocol, and later runs overwrite them.** The file name is built from task, speed and the flags `--dr --surfaces --pull`, but `PUSH_SCALE`, `DR_SCALE`, `DR_M/F/T`, `DR_OFF`, `TILES_FLAT` and `EVAL_NOEXIT` are environment variables and are not stored in the JSON. Observed consequences:
- `g2_s0/eval2_vel_0.3_dr_surf_pull.json` (fall 0.539) holds the load 0.75 run, not the full-load value 0.65 of S42, S43 and S47; the correct values are in `eval_g2s0_curve.txt` and `eval_g2s0_pull.txt`, which the site uses.
- `gfix_s0`, `gfix_track_s0`, `gfix_h16_s0` JSONs named `_dr_surf` come from evaluations before the S36 design-assignment fix (fall 0.644 for gfix_s0 against 0.295 in `eval_cmp.txt`); the site uses `eval_cmp.txt`.
- `g7_lqrres_s0/eval2_goal_goal_dr_surf.json` and `..._pull.json` were rewritten between 07:19 and 07:24 on 2026-11-07 by evaluations with other settings (undisturbed success went 0.470, 0.449, then 0.309; disturbed falls 0.819 then 0.901). The site now shows the values first read from the file (overall values equal S60 and S62, per-kind undisturbed values recovered, disturbed per-kind values lost), marked with a dagger. If the new files are a wide-randomisation evaluation, they will clash with the g9 evaluations as well.
- `wpperc_none_s0/eval_terrain.json` and the three `wpedge_*` files hold the 0.6 m/s evaluation; the 0.3 m/s blind row of S26 and the 0.3 m/s edge rows of S27 survive only in the log.
- Recommendation for the main session: store the environment settings in the JSON (`args` plus `env: {PUSH_SCALE, DR_SCALE, TILES_FLAT, EVAL_NOEXIT, DR_OFF, ...}`) and put a protocol tag in the file name before running the g9 evaluations; then run `python site/build_data.py --refresh`.

**F2. Number of worlds on the flat benchmark.** The log says 32 worlds per speed (S53 to S62). The files show the benchmark was run with `eval_designs = "nom,nom"`, so the `flat` row has 2 x 32 = 64 worlds (a fall rate of 0.047 is 3 of 64). The pooled `overall` row (all ten tile slots are flat there) has 640 worlds; the log values are the `flat` row. The GPU LQR calibration (S56) really has 32 worlds (0.12, 0.22, 0.97 are 4, 7 and 31 of 32). Pages and paper say 64 and 32.

**F3. Log discrepancies found.**
- S51 and S54 print ramp_up success 0.14 for `g2_goal_s0`; the data (and `eval_goal.txt`) give 0.18 (0.14 belongs to `g3_enc_s0`). The S54 range "0.14 to 0.20" should read 0.18 to 0.20.
- S56 says the LQR is "5 to 10 times more accurate" than the learned policies; with the S56 values and the S59 pure-learning error it is 5.7 times at 0.3, 10.4 at 0.6 and 16.7 at 0.9 m/s (the pages say 6 to 17).
- S60 says "40 percent faster" for the residual; with medians 2.23 and 2.21 s against 3.78 and 3.97 s it is 42 to 43 percent less time.
- The log and the paper give the continuous LQR gain as (0, 76.1, 4.33, 11.6); the exported model gives K = (-0.001, -76.14, -4.33, -11.58) for tau = -K s (magnitudes agree). The paper now states the signed values.
- The paper and story page said the reflected rotor inertia is 190 times the ball inertia and the unstable pole without damping 5.22 1/s; the parameters give 188.4 and 5.226 1/s. Both corrected (paper, story page).
- S34 and S35 numbers (per-design falls) check out against `gdiag_s0/falls_by.csv`.
- S43 table, S42 and S47 values check out against `eval_g2s0_curve.txt` and `eval_g2s0_pull.txt`, not against the JSONs (F1).

**F4. Data that exists only in the log (marked with a dagger on the pages).** S56 GPU LQR calibration (no output file: the script prints to the console), the S26 blind ridge row at 0.3 m/s, the S27 0.3 m/s rows, S30 throughput numbers, S31 and S32 ablation means, the S20 engine report numbers (in `experiments/mjx_port/REPORT.md`, partly in `bench_*.json`), the S15 to S17 CPU step and traction results (the CPU study tables come from `tracking_summary.csv`).

**F5. Excluded as superseded or invalid.** S33 and S35 DR rows, the "pushes limit" reading in S37 to S41 (kept only as the recipe-ceiling comparison, whose numbers are valid), S50 and S52, nan rows of S53 and S55 (replaced by S59), the S58 early checkpoint, `g4_pen_s0` (nan), and any result of the evaluation before the S36 fix.

**F6. Statements in the new text that are interpretations, not measurements.** Why authority 1.0 fails (S61 reasoning: the sum is clipped and exploration noise is of the same size); why authority 0.25 is worse (too little correction); why perception did not help (reward gives foresight no value, the 25 degree pitch hides the nearest floor, traction); these are phrased as probable and are not tested.

## Where the data is thin (candidates for more evaluations)

1. **Seeds.** Every row except the blind goal-only baseline (2 seeds), the LQR + residual rs 0.5 (2 seeds) and the g2_s0/g2_s1 velocity pair is single-seed. Perception arms, cross-focus, rs 0.25, rs 1.0, fall-penalty and tracking-weight rows need a second seed before a claim beyond "same band".
2. **Worlds per cell.** Flat benchmark 64 (LQR alone 32); terrain kind 176. Hard-kind differences of 0.05 are about 1.5 standard errors. A larger flat benchmark (for example 256 worlds) and a second evaluation seed would remove the 0.06 noise in the 0.9 m/s fall rates (0.27 against 0.16 between the two residual seeds).
3. **Held-out terrain and designs.** Training and evaluation use `tile_seed = 0` (the same 64 tiles, same severities) and the evaluation designs are the 11 training designs. Run `eval_tiles2.py --tile-seed 1 2 3` and add held-out designs (s0.5 and J0005 were dropped; tau3 and s1.5 were never in the 11).
4. **No CPU re-evaluation** of any tile-trained policy; the CPU cross-check only covers flat-ground policies (S21b, S23). A CPU tile or flat evaluation of `g7_lqrres_s0` would give the engine-gap figure for the headline policy.
5. **LQR alone on the tile and disturbance protocols.** S43 notes this calibration is missing: there is no LQR row for the velocity task on the tiles, the goal task or the disturbance load curve, so the residual's falls (0.63 against 0.38) cannot be compared with the LQR it is built on. `lqr_gpu.py` only runs the flat benchmark.
6. **Disturbance-load curve for the residual.** Only the two end points exist for `g5_mix_s0` and `g7_lqrres_s0/s1`; the full curve exists only for `g2_s0`.
7. **LQR calibration output file.** Save the S56 results as JSON (and run it with 64 worlds and two seeds like the learned rows).
8. **Perception.** One seed, goal-only, head mount only; no cones mount, no resolution sweep, no noise sweep, no depth student by distillation, no reward that values foresight.
9. **Videos.** The clip metadata does not record the randomisation setting or whether pushes were applied; `s46_track_push` falls after 0.94 s on a flat tile with unknown cause.
10. **Robustness (S63 onward).** `g9_dr_s0` and `g9_dr_s1` have no evaluation yet; `g8_lqrsoft_s0` is still training; no evaluation of `g7_lqrres` at a randomisation scale above 1 is stored under a name that can be told apart from the standard evaluation (F1).
11. **S61 interpretation** for authority 1.0 was not tested (for example with a smaller exploration log-std).

## Items to insert once the corrected-engine evaluations exist

- Run `eval_battery.py` for each `g12_*` run (and a pure-learning baseline, and the older policies for reference), then `python site/build_data.py`: Tables 2, 3 and 8 on `generalist.html` and the matching `paper_v2/tab_gen_*.tex` fill from `battery.json`; the g12 flat rows can be added to Table 1 by extending `spec` in `build_flat_table`.
- Replace the PENDING bullets and callouts in `generalist.html` chapters 3, 4, 5, 7 and 8, the red TBD marks in `paper_v2/sec_generalist.tex` and `main.tex` (abstract), and rewrite the conclusions from the new numbers.
- Re-measure on the corrected engine what was withdrawn: the recipe comparison and seed noise (two seeds), the disturbance-load curve, the perception arms (blind, scan, depth, frozen encoder, same recipe), goal-only and cross-focus, the residual authority sweep, and the randomisation ablation; add figures back by restoring the removed `fig_*` functions from the script history if wanted.
- Add held-out tile seeds and designs.

<!-- AUTO -->
## Automatic cross-check (written by site/build_data.py)

Supersession note (S92): the flat-benchmark rows printed in log entries S59 to S91 were measured on border tiles; every battery.json flat row was re-run on interior tiles, and the files are taken as correct. The re-run is a fresh sample, so its steady-state error and achieved speed also differ slightly from the logged values (by 0.001 to 0.01 m/s and up to 0.02 in ratio); the automatic checks therefore no longer compare flat rows with those entries. The LQR-alone row is the interior-tile re-measurement of the S92 addendum (log values, dagger); the zero-shot arrestor flat row was not re-run and is shown with a caveat.

1075 quantities compared: 1070 match the study log within rounding, 5 differ.

### Differences

| entry | where | quantity | data file | log | note |
|---|---|---|---|---|---|
| paper Prop. 1 | Setup page / paper | unstable pole without damping | 5.2261 | 5.22 |  |
| paper | Setup page / paper | J_eff / ball inertia | 188.4407 | 190 |  |
| S92 addendum | Table 1 LQR alone | falls @0.9 | 0.9100 | 0.89 | lqr_flat_interior.log prints 0.91 (error 0.026, achieved 1.00 over survivors); the S92 addendum text says 0.89; the file is used |
| S95 addendum | text | 50 percent point, largest (fraction of r) | 0.2884 | 0.22 | the log gives 'roughly 0.18 to 0.22 r' for seed 0 at friction 0.001; here the crossing is found by linear interpolation of the pooled 0.5 cm-bin curves per size and feature (eight curves) at friction 0.01; the pages quote the computed range and mean |
| S95 addendum | text | 50 percent point, mean (fraction of r) | 0.2364 | 0.2 | the log gives 'roughly 0.18 to 0.22 r' for seed 0 at friction 0.001; here the crossing is found by linear interpolation of the pooled 0.5 cm-bin curves per size and feature (eight curves) at friction 0.01; the pages quote the computed range and mean |

### Statements checked against the source code

| statement on the pages | pattern searched | file | found |
|---|---|---|---|
| pull force 3 to 12 N | `lo=3.0, hi=12.0` | experiments/warp_ppo/train_dev.py | yes |
| pull duration 50 to 199 steps | `torch.randint(50, 200` | experiments/warp_ppo/train_dev.py | yes |
| pull interval 400 to 1000 steps | `lo=400, hi=1000` | experiments/warp_ppo/train_dev.py | yes |
| push 5 control steps | `torch.full_like(self.push_left, 5)` | experiments/warp_ppo/train_dev.py | yes |
| push interval 200 to 400 steps | `lo=200, hi=400` | experiments/warp_ppo/train_dev.py | yes |
| surface friction jitter 0.15 | `lo=-0.15, hi=0.15` | experiments/warp_ppo/train_dev.py | yes |
| max latency prob 0.6 | `min(self.drc.get('latency_p', 0.2) * self.dr_l, 0.6)` | experiments/warp_ppo/train_dev.py | yes |
| start offset 0.3 m | `lo=-0.3, hi=0.3` | experiments/warp_ppo/train_dev.py | yes |
| master scale touches six ranges only | `self.dr_m = self.dr_f = self.dr_t = self.dr_d = self.dr_l = self.dr_n = float(x)` | experiments/warp_ppo/train_dev.py | yes |

### All matches

| log entry | where used | quantity | value from data file | value in log | tolerance |
|---|---|---|---|---|---|
| S-model, paper Prop. 1 | Setup page / paper | unstable pole with damping | 4.7900 | 4.79 | 0.006 |
| paper Prop. 1 | Setup page / paper | unstable pole without damping | 5.2261 | 5.22 | 0.006 |
| S-model | Setup page / paper | NMP zero | 5.7821 | 5.78 | 0.006 |
| S-model | Setup page / paper | kappa [rad per m/s^2] | 0.1765 | 0.176 | 0.001 |
| S-model | Setup page / paper | J_eff | 0.1025 | 0.1025 | 0.0006 |
| paper | Setup page / paper | J_eff / ball inertia | 188.4407 | 190 | 1.5 |
| S-model (log gives the magnitude) | Setup page / paper | continuous K theta | 76.1418 | 76.1 | 0.06 |
| S-model | Setup page / paper | continuous K phidot | 4.3329 | 4.33 | 0.006 |
| S-model | Setup page / paper | continuous K thetadot | 11.5814 | 11.6 | 0.06 |
| paper | Setup page / paper | closed-loop fast pole | -20.3263 | -20.3 | 0.06 |
| config | Table D | mass range | 0.1500 | 0.15 | 1e-09 |
| config | Table D | friction range | 0.2000 | 0.2 | 1e-09 |
| config | Table D | damping lo | 0.0700 | 0.07 | 1e-09 |
| config | Table D | damping hi | 0.1500 | 0.15 | 1e-09 |
| config | Table D | torque range | 0.1500 | 0.15 | 1e-09 |
| config | Table D | latency p | 0.2000 | 0.2 | 1e-09 |
| config | Table D | obs noise | 0.0100 | 0.01 | 1e-09 |
| config | Table D | push min | 5.0000 | 5.0 | 1e-09 |
| config | Table D | push max | 20.0000 | 20.0 | 1e-09 |
| config | Table D | push every lo | 2.0000 | 2.0 | 1e-09 |
| config | Table D | push every hi | 4.0000 | 4.0 | 1e-09 |
| config | Table D | push dur | 0.0500 | 0.05 | 1e-09 |
| config | Method text | cmd_vmax | 1.2000 | 1.2 | 1e-09 |
| config | Method text | cmd_wmax | 1.0000 | 1.0 | 1e-09 |
| config | Method text | action_repeat | 5.0000 | 5 | 1e-09 |
| config | Method text | max episode steps | 800.0000 | 800 | 1e-09 |
| config | Method text | tilt limit | 70.0000 | 70 | 1e-09 |
| config | Method text | ball offset limit | 30.0000 | 30 | 1e-09 |
| config | Method text | cmd resample steps | 300.0000 | 300 | 1e-09 |
| config | Method text | cmd zero prob | 0.1000 | 0.1 | 1e-09 |
| config | Method text | torque limit | 30.0000 | 30.0 | 1e-09 |
| S72-S74 | Table 2/4 g12_res_s0 | goal success undisturbed DR 1 | 0.5807 | 0.58 | 0.006 |
| S72-S74 | Table 2/4 g12_res_s0 | goal success disturbed DR 1 | 0.2540 | 0.25 | 0.006 |
| S72-S74 | Table 2/4 g12_res_s0 | goal success undisturbed DR 1.75 | 0.5233 | 0.52 | 0.006 |
| S72-S74 | Table 2/4 g12_res_s0 | goal success disturbed DR 1.75 | 0.2267 | 0.23 | 0.006 |
| S72-S74 | Table 2/4 g12_res_s0 | goal success undisturbed DR 2.5 | 0.4489 | 0.45 | 0.006 |
| S72-S74 | Table 2/4 g12_res_s0 | goal success disturbed DR 2.5 | 0.1795 | 0.18 | 0.006 |
| S72-S74 | Table 2 g12_res_s0 | median time | 2.6100 | 2.61 | 0.13 |
| S72/S73 | Table 2 g12_res_s0 | step_up | 0.3295 | 0.33 | 0.006 |
| S72/S73 | Table 2 g12_res_s0 | stairs | 0.3750 | 0.38 | 0.006 |
| S72/S73 | Table 2 g12_res_s0 | bump | 0.1705 | 0.17 | 0.006 |
| S72/S73 | Table 2 g12_res_s0 | ramp_up | 0.7443 | 0.74 | 0.006 |
| S72-S74 | Table 3 g12_res_s0 | vel falls undisturbed | 0.5540 | 0.55 | 0.006 |
| S72-S74 | Table 3 g12_res_s0 | vel falls disturbed | 0.7222 | 0.72 | 0.006 |
| S72-S74 | Table 2/4 g12_dr175_s0 | goal success undisturbed DR 1 | 0.5841 | 0.58 | 0.006 |
| S72-S74 | Table 2/4 g12_dr175_s0 | goal success disturbed DR 1 | 0.2858 | 0.29 | 0.006 |
| S72-S74 | Table 2/4 g12_dr175_s0 | goal success undisturbed DR 1.75 | 0.5352 | 0.54 | 0.006 |
| S72-S74 | Table 2/4 g12_dr175_s0 | goal success disturbed DR 1.75 | 0.2455 | 0.25 | 0.006 |
| S72-S74 | Table 2/4 g12_dr175_s0 | goal success undisturbed DR 2.5 | 0.4517 | 0.45 | 0.006 |
| S72-S74 | Table 2/4 g12_dr175_s0 | goal success disturbed DR 2.5 | 0.1909 | 0.19 | 0.006 |
| S72-S74 | Table 2 g12_dr175_s0 | median time | 2.4400 | 2.44 | 0.13 |
| S72/S73 | Table 2 g12_dr175_s0 | step_up | 0.3125 | 0.31 | 0.006 |
| S72/S73 | Table 2 g12_dr175_s0 | stairs | 0.4375 | 0.44 | 0.006 |
| S72/S73 | Table 2 g12_dr175_s0 | bump | 0.2614 | 0.26 | 0.006 |
| S72/S73 | Table 2 g12_dr175_s0 | ramp_up | 0.6534 | 0.65 | 0.006 |
| S72-S74 | Table 3 g12_dr175_s0 | vel falls undisturbed | 0.5369 | 0.54 | 0.006 |
| S72-S74 | Table 3 g12_dr175_s0 | vel falls disturbed | 0.7148 | 0.71 | 0.006 |
| S72-S74 | Table 2/4 g12_soft175_s0 | goal success undisturbed DR 1 | 0.6301 | 0.63 | 0.006 |
| S72-S74 | Table 2/4 g12_soft175_s0 | goal success disturbed DR 1 | 0.3301 | 0.33 | 0.006 |
| S72-S74 | Table 2/4 g12_soft175_s0 | goal success undisturbed DR 1.75 | 0.5727 | 0.57 | 0.006 |
| S72-S74 | Table 2/4 g12_soft175_s0 | goal success disturbed DR 1.75 | 0.2858 | 0.29 | 0.006 |
| S72-S74 | Table 2/4 g12_soft175_s0 | goal success undisturbed DR 2.5 | 0.4943 | 0.49 | 0.006 |
| S72-S74 | Table 2/4 g12_soft175_s0 | goal success disturbed DR 2.5 | 0.2330 | 0.23 | 0.006 |
| S72-S74 | Table 2 g12_soft175_s0 | median time | 3.0600 | 3.06 | 0.13 |
| S72/S73 | Table 2 g12_soft175_s0 | step_up | 0.3466 | 0.35 | 0.006 |
| S72/S73 | Table 2 g12_soft175_s0 | stairs | 0.3920 | 0.39 | 0.006 |
| S72/S73 | Table 2 g12_soft175_s0 | bump | 0.1591 | 0.16 | 0.006 |
| S72/S73 | Table 2 g12_soft175_s0 | ramp_up | 0.7500 | 0.75 | 0.006 |
| S72-S74 | Table 3 g12_soft175_s0 | vel falls undisturbed | 0.5460 | 0.55 | 0.006 |
| S72-S74 | Table 3 g12_soft175_s0 | vel falls disturbed | 0.7142 | 0.71 | 0.006 |
| S72-S74 | Table 2/4 g12_res_s1 | goal success undisturbed DR 1 | 0.5778 | 0.58 | 0.006 |
| S72-S74 | Table 2/4 g12_res_s1 | goal success disturbed DR 1 | 0.2659 | 0.27 | 0.006 |
| S72-S74 | Table 2/4 g12_res_s1 | goal success undisturbed DR 1.75 | 0.5432 | 0.54 | 0.006 |
| S72-S74 | Table 2/4 g12_res_s1 | goal success disturbed DR 1.75 | 0.2455 | 0.25 | 0.006 |
| S72-S74 | Table 2/4 g12_res_s1 | goal success undisturbed DR 2.5 | 0.4415 | 0.44 | 0.006 |
| S72-S74 | Table 2/4 g12_res_s1 | goal success disturbed DR 2.5 | 0.1909 | 0.19 | 0.006 |
| S72-S74 | Table 2 g12_res_s1 | median time | 2.4300 | 2.43 | 0.13 |
| S72/S73 | Table 2 g12_res_s1 | step_up | 0.2841 | 0.28 | 0.006 |
| S72/S73 | Table 2 g12_res_s1 | stairs | 0.3864 | 0.39 | 0.006 |
| S72/S73 | Table 2 g12_res_s1 | bump | 0.1989 | 0.2 | 0.006 |
| S72/S73 | Table 2 g12_res_s1 | ramp_up | 0.6875 | 0.69 | 0.006 |
| S72-S74 | Table 3 g12_res_s1 | vel falls undisturbed | 0.5534 | 0.55 | 0.006 |
| S72-S74 | Table 3 g12_res_s1 | vel falls disturbed | 0.7415 | 0.74 | 0.006 |
| S72-S74 | Table 2/4 g12_dr175_s1 | goal success undisturbed DR 1 | 0.5994 | 0.6 | 0.006 |
| S72-S74 | Table 2/4 g12_dr175_s1 | goal success disturbed DR 1 | 0.2983 | 0.3 | 0.006 |
| S72-S74 | Table 2/4 g12_dr175_s1 | goal success undisturbed DR 1.75 | 0.5557 | 0.56 | 0.006 |
| S72-S74 | Table 2/4 g12_dr175_s1 | goal success disturbed DR 1.75 | 0.2460 | 0.25 | 0.006 |
| S72-S74 | Table 2/4 g12_dr175_s1 | goal success undisturbed DR 2.5 | 0.4716 | 0.47 | 0.006 |
| S72-S74 | Table 2/4 g12_dr175_s1 | goal success disturbed DR 2.5 | 0.1949 | 0.19 | 0.006 |
| S72-S74 | Table 2 g12_dr175_s1 | median time | 2.3300 | 2.3 | 0.13 |
| S72-S74 | Table 3 g12_dr175_s1 | vel falls undisturbed | 0.5267 | 0.53 | 0.006 |
| S72-S74 | Table 3 g12_dr175_s1 | vel falls disturbed | 0.7148 | 0.71 | 0.006 |
| S72-S74 | Table 2/4 g12_soft175_s1 | goal success undisturbed DR 1 | 0.6216 | 0.62 | 0.006 |
| S72-S74 | Table 2/4 g12_soft175_s1 | goal success disturbed DR 1 | 0.3403 | 0.34 | 0.006 |
| S72-S74 | Table 2/4 g12_soft175_s1 | goal success undisturbed DR 1.75 | 0.5818 | 0.58 | 0.006 |
| S72-S74 | Table 2/4 g12_soft175_s1 | goal success disturbed DR 1.75 | 0.2892 | 0.29 | 0.006 |
| S72-S74 | Table 2/4 g12_soft175_s1 | goal success undisturbed DR 2.5 | 0.5165 | 0.52 | 0.006 |
| S72-S74 | Table 2/4 g12_soft175_s1 | goal success disturbed DR 2.5 | 0.2392 | 0.24 | 0.006 |
| S72-S74 | Table 2 g12_soft175_s1 | median time | 3.1200 | 3.0 | 0.13 |
| S72-S74 | Table 3 g12_soft175_s1 | vel falls undisturbed | 0.5290 | 0.53 | 0.006 |
| S72-S74 | Table 3 g12_soft175_s1 | vel falls disturbed | 0.7256 | 0.73 | 0.006 |
| S72-S74 | Table 2/4 g13_softrs075_s0 | goal success undisturbed DR 1 | 0.5801 | 0.58 | 0.006 |
| S72-S74 | Table 2/4 g13_softrs075_s0 | goal success disturbed DR 1 | 0.3182 | 0.32 | 0.006 |
| S72-S74 | Table 2/4 g13_softrs075_s0 | goal success undisturbed DR 1.75 | 0.5261 | 0.53 | 0.006 |
| S72-S74 | Table 2/4 g13_softrs075_s0 | goal success disturbed DR 1.75 | 0.2642 | 0.26 | 0.006 |
| S72-S74 | Table 2/4 g13_softrs075_s0 | goal success undisturbed DR 2.5 | 0.4494 | 0.45 | 0.006 |
| S72-S74 | Table 2/4 g13_softrs075_s0 | goal success disturbed DR 2.5 | 0.2108 | 0.21 | 0.006 |
| S72-S74 | Table 2 g13_softrs075_s0 | median time | 3.3700 | 3.3 | 0.13 |
| S72-S74 | Table 3 g13_softrs075_s0 | vel falls undisturbed | 0.5369 | 0.54 | 0.006 |
| S72-S74 | Table 3 g13_softrs075_s0 | vel falls disturbed | 0.6989 | 0.7 | 0.006 |
| S72-S74 | Table 2/4 g12_pure_s0 | goal success undisturbed DR 1 | 0.4420 | 0.44 | 0.006 |
| S72-S74 | Table 2/4 g12_pure_s0 | goal success disturbed DR 1 | 0.2636 | 0.26 | 0.006 |
| S72-S74 | Table 2/4 g12_pure_s0 | goal success undisturbed DR 1.75 | 0.4284 | 0.43 | 0.006 |
| S72-S74 | Table 2/4 g12_pure_s0 | goal success disturbed DR 1.75 | 0.2256 | 0.23 | 0.006 |
| S72-S74 | Table 2/4 g12_pure_s0 | goal success undisturbed DR 2.5 | 0.3739 | 0.37 | 0.006 |
| S72-S74 | Table 2/4 g12_pure_s0 | goal success disturbed DR 2.5 | 0.1824 | 0.18 | 0.006 |
| S72-S74 | Table 2 g12_pure_s0 | median time | 3.0800 | 3.1 | 0.13 |
| S72-S74 | Table 3 g12_pure_s0 | vel falls undisturbed | 0.3193 | 0.32 | 0.006 |
| S72-S74 | Table 3 g12_pure_s0 | vel falls disturbed | 0.5364 | 0.54 | 0.006 |
| S73 | text | res s0 vel err undisturbed | 0.1121 | 0.112 | 0.0006 |
| S73 | text | res s1 vel err disturbed | 0.2051 | 0.205 | 0.0006 |
| S76 | Table 2/4 g14_fp5_s0 | goal undisturbed DR 1 | 0.5716 | 0.57 | 0.006 |
| S76 | Table 2/4 g14_fp5_s0 | goal disturbed DR 1 | 0.2705 | 0.27 | 0.006 |
| S76 | Table 2/4 g14_fp5_s0 | goal undisturbed DR 1.75 | 0.5318 | 0.53 | 0.006 |
| S76 | Table 2/4 g14_fp5_s0 | goal disturbed DR 1.75 | 0.2472 | 0.25 | 0.006 |
| S76 | Table 2/4 g14_fp5_s0 | goal undisturbed DR 2.5 | 0.4574 | 0.46 | 0.006 |
| S76 | Table 2/4 g14_fp5_s0 | goal disturbed DR 2.5 | 0.1767 | 0.18 | 0.006 |
| S76 | Table 3 g14_fp5_s0 | vel falls undisturbed | 0.5364 | 0.54 | 0.006 |
| S76 | Table 3 g14_fp5_s0 | vel falls disturbed | 0.7330 | 0.73 | 0.006 |
| S76 | Table 2/4 g14_softfp5_s0 | goal undisturbed DR 1 | 0.6267 | 0.63 | 0.006 |
| S76 | Table 2/4 g14_softfp5_s0 | goal disturbed DR 1 | 0.3256 | 0.33 | 0.006 |
| S76 | Table 2/4 g14_softfp5_s0 | goal undisturbed DR 1.75 | 0.5875 | 0.59 | 0.006 |
| S76 | Table 2/4 g14_softfp5_s0 | goal disturbed DR 1.75 | 0.2727 | 0.27 | 0.006 |
| S76 | Table 2/4 g14_softfp5_s0 | goal undisturbed DR 2.5 | 0.4989 | 0.5 | 0.006 |
| S76 | Table 2/4 g14_softfp5_s0 | goal disturbed DR 2.5 | 0.2165 | 0.22 | 0.006 |
| S76 | Table 3 g14_softfp5_s0 | vel falls undisturbed | 0.5364 | 0.54 | 0.006 |
| S76 | Table 3 g14_softfp5_s0 | vel falls disturbed | 0.7131 | 0.71 | 0.006 |
| S76 | Table 2/4 g14_pramp_s0 | goal undisturbed DR 1 | 0.5687 | 0.57 | 0.006 |
| S76 | Table 2/4 g14_pramp_s0 | goal disturbed DR 1 | 0.2472 | 0.25 | 0.006 |
| S76 | Table 2/4 g14_pramp_s0 | goal undisturbed DR 1.75 | 0.5176 | 0.52 | 0.006 |
| S76 | Table 2/4 g14_pramp_s0 | goal disturbed DR 1.75 | 0.2068 | 0.21 | 0.006 |
| S76 | Table 2/4 g14_pramp_s0 | goal undisturbed DR 2.5 | 0.4659 | 0.47 | 0.006 |
| S76 | Table 2/4 g14_pramp_s0 | goal disturbed DR 2.5 | 0.1619 | 0.16 | 0.006 |
| S76 | Table 3 g14_pramp_s0 | vel falls undisturbed | 0.5409 | 0.54 | 0.006 |
| S76 | Table 3 g14_pramp_s0 | vel falls disturbed | 0.7290 | 0.73 | 0.006 |
| S76 | Table 5 | g12_dr175_s0 held-out undisturbed | 0.5222 | 0.52 | 0.006 |
| S76 | Table 5 | g12_dr175_s0 held-out disturbed | 0.2597 | 0.26 | 0.006 |
| S76 | Table 5 | g12_dr175_s0 held-out time undisturbed | 2.4400 | 2.4 | 0.06 |
| S76 | Table 5 | g12_dr175_s0 held-out time disturbed | 2.3500 | 2.4 | 0.06 |
| S76 | Table 5 | g12_dr175_s1 held-out undisturbed | 0.5330 | 0.53 | 0.006 |
| S76 | Table 5 | g12_dr175_s1 held-out disturbed | 0.2631 | 0.26 | 0.006 |
| S76 | Table 5 | g12_dr175_s1 held-out time undisturbed | 2.3200 | 2.3 | 0.06 |
| S76 | Table 5 | g12_dr175_s1 held-out time disturbed | 2.2400 | 2.2 | 0.06 |
| S78 | Table 13 g15_edge_depth_s0 | goal undisturbed DR 1 | 0.6040 | 0.6 | 0.006 |
| S78 | Table 11 g15_edge_depth_s0 | goal disturbed DR 1 | 0.2903 | 0.29 | 0.006 |
| S78 | Table 13 g15_edge_depth_s0 | goal undisturbed DR 1.75 | 0.5312 | 0.53 | 0.006 |
| S78 | Table 11 g15_edge_depth_s0 | goal disturbed DR 1.75 | 0.2557 | 0.26 | 0.006 |
| S78 | Table 13 g15_edge_depth_s0 | goal undisturbed DR 2.5 | 0.4523 | 0.45 | 0.006 |
| S78 | Table 11 g15_edge_depth_s0 | goal disturbed DR 2.5 | 0.1767 | 0.18 | 0.006 |
| S78 | Table 11 g15_edge_depth_s0 | median time | 2.3600 | 2.36 | 0.06 |
| S78 | Table 11 g15_edge_depth_s0 | vel falls undisturbed | 0.5648 | 0.56 | 0.006 |
| S78 | Table 11 g15_edge_depth_s0 | vel falls disturbed | 0.7301 | 0.73 | 0.006 |
| S78 | Table 13 g15_edge_blind_s0 | goal undisturbed DR 1 | 0.5966 | 0.6 | 0.006 |
| S78 | Table 11 g15_edge_blind_s0 | goal disturbed DR 1 | 0.2767 | 0.28 | 0.006 |
| S78 | Table 13 g15_edge_blind_s0 | goal undisturbed DR 1.75 | 0.5381 | 0.54 | 0.006 |
| S78 | Table 11 g15_edge_blind_s0 | goal disturbed DR 1.75 | 0.2261 | 0.23 | 0.006 |
| S78 | Table 13 g15_edge_blind_s0 | goal undisturbed DR 2.5 | 0.4688 | 0.47 | 0.006 |
| S78 | Table 11 g15_edge_blind_s0 | goal disturbed DR 2.5 | 0.1869 | 0.19 | 0.006 |
| S78 | Table 11 g15_edge_blind_s0 | median time | 2.3000 | 2.3 | 0.06 |
| S78 | Table 11 g15_edge_blind_s0 | vel falls undisturbed | 0.5290 | 0.53 | 0.006 |
| S78 | Table 11 g15_edge_blind_s0 | vel falls disturbed | 0.7301 | 0.73 | 0.006 |
| S79 | Table 6 | u fall rate | 0.4256 | 0.43 | 0.006 |
| S79 | Table 6 | u offset-ended share | 0.9519 | 0.95 | 0.006 |
| S79 | Table 6 | u saturation percent | 0.7397 | 0.7 | 0.06 |
| S79 | Table 6 | u derailed first | 0.7784 | 0.78 | 0.006 |
| S79 | Table 6 | u median tilt at derail | 14.6517 | 14.7 | 0.06 |
| S79 | Table 6 | d fall rate | 0.7199 | 0.72 | 0.006 |
| S79 | Table 6 | d offset-ended share | 0.9700 | 0.97 | 0.006 |
| S79 | Table 6 | d saturation percent | 0.4128 | 0.4 | 0.06 |
| S79 | Table 6 | d derailed first | 0.4854 | 0.49 | 0.006 |
| S79 | Table 6 | d median tilt at derail | 15.4053 | 15.4 | 0.06 |
| S79 | Table 6 | lqr fall rate | 0.8523 | 0.85 | 0.006 |
| S79 | Table 6 | lqr offset-ended share | 0.9833 | 0.98 | 0.006 |
| S79 | Table 6 | lqr saturation percent | 0.4282 | 0.4 | 0.06 |
| S79 | text | pulls active in disturbed falls | 0.5809 | 0.58 | 0.006 |
| S79 | Table 8 | none success undisturbed | 0.5744 | 0.57 | 0.006 |
| S79 | Table 8 | none success disturbed | 0.2795 | 0.28 | 0.006 |
| S79 | Table 8 | 135 success undisturbed | 0.6631 | 0.66 | 0.006 |
| S79 | Table 8 | 135 success disturbed | 0.3693 | 0.37 | 0.006 |
| S79 | Table 8 | 120 success undisturbed | 0.6937 | 0.69 | 0.006 |
| S79 | Table 8 | 120 success disturbed | 0.3886 | 0.39 | 0.006 |
| S79 | Table 8 | 110 success undisturbed | 0.7233 | 0.72 | 0.006 |
| S79 | Table 8 | 110 success disturbed | 0.4034 | 0.4 | 0.006 |
| S79 | Table 8 | 100 success undisturbed | 0.7455 | 0.75 | 0.006 |
| S79 | Table 8 | 100 success disturbed | 0.4477 | 0.45 | 0.006 |
| S79 | Table 8 | 95 success undisturbed | 0.7506 | 0.75 | 0.006 |
| S79 | Table 8 | 95 success disturbed | 0.4341 | 0.43 | 0.006 |
| S79 | Table 8 | 90 success undisturbed | 0.7568 | 0.76 | 0.006 |
| S79 | Table 8 | 90 success disturbed | 0.4369 | 0.44 | 0.006 |
| S79 | Table 8 | clearance at 100 deg [cm] | 6.4982 | 6.5 | 0.08 |
| S79 | Table 8 | clearance at 120 deg [cm] | 3.3000 | 3.3 | 0.08 |
| S79 | Table 8 | clearance at 135 deg [cm] | 1.2704 | 1.2 | 0.08 |
| S79 | Table 9 | friction 0 undisturbed | 0.7455 | 0.75 | 0.006 |
| S79 | Table 9 | friction 0 disturbed | 0.4477 | 0.45 | 0.006 |
| S79 | Table 9 | friction 0.05 undisturbed | 0.7426 | 0.74 | 0.006 |
| S79 | Table 9 | friction 0.05 disturbed | 0.4222 | 0.42 | 0.006 |
| S79 | Table 9 | friction 0.2 undisturbed | 0.7574 | 0.76 | 0.006 |
| S79 | Table 9 | friction 0.2 disturbed | 0.4170 | 0.42 | 0.006 |
| S80 | Table 10 g16_ret100_s0 | goal undisturbed DR 1 | 0.7125 | 0.71 | 0.006 |
| S80 | Table 10 g16_ret100_s0 | goal disturbed DR 1 | 0.4625 | 0.46 | 0.006 |
| S80 | Table 10 g16_ret100_s0 | goal undisturbed DR 1.75 | 0.7028 | 0.7 | 0.006 |
| S80 | Table 10 g16_ret100_s0 | goal disturbed DR 1.75 | 0.4210 | 0.42 | 0.006 |
| S80 | Table 10 g16_ret100_s0 | goal undisturbed DR 2.5 | 0.6153 | 0.62 | 0.006 |
| S80 | Table 10 g16_ret100_s0 | goal disturbed DR 2.5 | 0.3523 | 0.35 | 0.006 |
| S80 | Table 10 g16_ret100_s0 | vel falls undisturbed | 0.3119 | 0.31 | 0.006 |
| S80 | Table 10 g16_ret100_s0 | vel falls disturbed | 0.4875 | 0.49 | 0.006 |
| S80 | Table 10 g16_ret100_s1 | goal undisturbed DR 1 | 0.7631 | 0.76 | 0.006 |
| S80 | Table 10 g16_ret100_s1 | goal disturbed DR 1 | 0.4852 | 0.49 | 0.006 |
| S80 | Table 10 g16_ret100_s1 | goal undisturbed DR 1.75 | 0.7330 | 0.73 | 0.006 |
| S80 | Table 10 g16_ret100_s1 | goal disturbed DR 1.75 | 0.4358 | 0.44 | 0.006 |
| S80 | Table 10 g16_ret100_s1 | goal undisturbed DR 2.5 | 0.6727 | 0.67 | 0.006 |
| S80 | Table 10 g16_ret100_s1 | goal disturbed DR 2.5 | 0.3716 | 0.37 | 0.006 |
| S80 | Table 10 g16_ret100_s1 | vel falls undisturbed | 0.3250 | 0.32 | 0.006 |
| S80 | Table 10 g16_ret100_s1 | vel falls disturbed | 0.4960 | 0.5 | 0.006 |
| S80 | Table 10 g12_pure_s1 | goal undisturbed DR 1 | 0.5136 | 0.51 | 0.006 |
| S80 | Table 10 g12_pure_s1 | goal disturbed DR 1 | 0.2693 | 0.27 | 0.006 |
| S80 | Table 10 g12_pure_s1 | goal undisturbed DR 1.75 | 0.4449 | 0.44 | 0.006 |
| S80 | Table 10 g12_pure_s1 | goal disturbed DR 1.75 | 0.2341 | 0.23 | 0.006 |
| S80 | Table 10 g12_pure_s1 | goal undisturbed DR 2.5 | 0.3858 | 0.39 | 0.006 |
| S80 | Table 10 g12_pure_s1 | goal disturbed DR 2.5 | 0.1926 | 0.19 | 0.006 |
| S80 | Table 10 g12_pure_s1 | vel falls undisturbed | 0.3756 | 0.38 | 0.006 |
| S80 | Table 10 g12_pure_s1 | vel falls disturbed | 0.5722 | 0.57 | 0.006 |
| S80 | Table 10 zero-shot | vel falls undisturbed | 0.3619 | 0.36 | 0.006 |
| S80 | Table 10 zero-shot | vel falls disturbed | 0.5330 | 0.53 | 0.006 |
| S80/S82 | Table 12 | g18_ret_rs025_s0 held-out undisturbed | 0.6574 | 0.66 | 0.006 |
| S80/S82 | Table 12 | g18_ret_rs025_s0 held-out disturbed | 0.3830 | 0.38 | 0.006 |
| S80/S82 | Table 12 | g18_ret_rs025_s1 held-out undisturbed | 0.6869 | 0.69 | 0.006 |
| S80/S82 | Table 12 | g18_ret_rs025_s1 held-out disturbed | 0.3943 | 0.39 | 0.006 |
| S80/S82 | Table 12 | g16_ret100_s0 held-out undisturbed | 0.6591 | 0.66 | 0.006 |
| S80/S82 | Table 12 | g16_ret100_s0 held-out disturbed | 0.4199 | 0.42 | 0.006 |
| S80/S82 | Table 12 | g16_ret100_s1 held-out undisturbed | 0.6818 | 0.68 | 0.006 |
| S80/S82 | Table 12 | g16_ret100_s1 held-out disturbed | 0.4312 | 0.43 | 0.006 |
| S80/S82 | Table 12 | g16_pure_ret100_s0 held-out undisturbed | 0.6625 | 0.66 | 0.006 |
| S80/S82 | Table 12 | g16_pure_ret100_s0 held-out disturbed | 0.4648 | 0.46 | 0.006 |
| S80/S82 | Table 12 | g16_pure_ret100_s1 held-out undisturbed | 0.7102 | 0.71 | 0.006 |
| S80/S82 | Table 12 | g16_pure_ret100_s1 held-out disturbed | 0.4744 | 0.47 | 0.006 |
| S81-S83 | Table 11 g16_pure_ret100_s0 | goal undisturbed DR 1 | 0.6955 | 0.7 | 0.006 |
| S81-S83 | Table 11 g16_pure_ret100_s0 | goal disturbed DR 1 | 0.4955 | 0.5 | 0.006 |
| S81-S83 | Table 11 g16_pure_ret100_s0 | goal undisturbed DR 1.75 | 0.6739 | 0.67 | 0.006 |
| S81-S83 | Table 11 g16_pure_ret100_s0 | goal disturbed DR 1.75 | 0.4494 | 0.45 | 0.006 |
| S81-S83 | Table 11 g16_pure_ret100_s0 | goal undisturbed DR 2.5 | 0.6420 | 0.64 | 0.006 |
| S81-S83 | Table 11 g16_pure_ret100_s0 | goal disturbed DR 2.5 | 0.4006 | 0.4 | 0.006 |
| S81-S83 | Table 11 g16_pure_ret100_s0 | vel falls undisturbed | 0.1625 | 0.16 | 0.006 |
| S81-S83 | Table 11 g16_pure_ret100_s0 | vel falls disturbed | 0.3438 | 0.34 | 0.006 |
| S81-S83 | Table 11 g17_ret100_trk3_s0 | goal undisturbed DR 1 | 0.7080 | 0.71 | 0.006 |
| S81-S83 | Table 11 g17_ret100_trk3_s0 | goal disturbed DR 1 | 0.4892 | 0.49 | 0.006 |
| S81-S83 | Table 11 g17_ret100_trk3_s0 | goal undisturbed DR 1.75 | 0.6847 | 0.68 | 0.006 |
| S81-S83 | Table 11 g17_ret100_trk3_s0 | goal disturbed DR 1.75 | 0.4301 | 0.43 | 0.006 |
| S81-S83 | Table 11 g17_ret100_trk3_s0 | goal undisturbed DR 2.5 | 0.6159 | 0.62 | 0.006 |
| S81-S83 | Table 11 g17_ret100_trk3_s0 | goal disturbed DR 2.5 | 0.3568 | 0.36 | 0.006 |
| S81-S83 | Table 11 g17_ret100_trk3_s0 | vel falls undisturbed | 0.3182 | 0.32 | 0.006 |
| S81-S83 | Table 11 g17_ret100_trk3_s0 | vel falls disturbed | 0.4602 | 0.46 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs025_s0 | goal undisturbed DR 1 | 0.7148 | 0.71 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs025_s0 | goal disturbed DR 1 | 0.4182 | 0.42 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs025_s0 | goal undisturbed DR 1.75 | 0.6801 | 0.68 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs025_s0 | goal disturbed DR 1.75 | 0.3625 | 0.36 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs025_s0 | goal undisturbed DR 2.5 | 0.6040 | 0.6 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs025_s0 | goal disturbed DR 2.5 | 0.2909 | 0.29 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs025_s0 | vel falls undisturbed | 0.3398 | 0.34 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs025_s0 | vel falls disturbed | 0.5523 | 0.55 | 0.006 |
| S81-S83 | Table 11 g18_ret_soft_s0 | goal undisturbed DR 1 | 0.7534 | 0.75 | 0.006 |
| S81-S83 | Table 11 g18_ret_soft_s0 | goal disturbed DR 1 | 0.5080 | 0.51 | 0.006 |
| S81-S83 | Table 11 g18_ret_soft_s0 | goal undisturbed DR 1.75 | 0.7460 | 0.75 | 0.006 |
| S81-S83 | Table 11 g18_ret_soft_s0 | goal disturbed DR 1.75 | 0.4688 | 0.47 | 0.006 |
| S81-S83 | Table 11 g18_ret_soft_s0 | goal undisturbed DR 2.5 | 0.6614 | 0.66 | 0.006 |
| S81-S83 | Table 11 g18_ret_soft_s0 | goal disturbed DR 2.5 | 0.3898 | 0.39 | 0.006 |
| S81-S83 | Table 11 g18_ret_soft_s0 | vel falls undisturbed | 0.2932 | 0.29 | 0.006 |
| S81-S83 | Table 11 g18_ret_soft_s0 | vel falls disturbed | 0.4875 | 0.49 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs015_s0 | goal undisturbed DR 1 | 0.6881 | 0.69 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs015_s0 | goal disturbed DR 1 | 0.3534 | 0.35 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs015_s0 | goal undisturbed DR 1.75 | 0.6477 | 0.65 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs015_s0 | goal disturbed DR 1.75 | 0.3108 | 0.31 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs015_s0 | goal undisturbed DR 2.5 | 0.5580 | 0.56 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs015_s0 | goal disturbed DR 2.5 | 0.2176 | 0.22 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs025_s1 | goal undisturbed DR 1 | 0.7619 | 0.76 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs025_s1 | goal disturbed DR 1 | 0.4324 | 0.43 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs025_s1 | goal undisturbed DR 1.75 | 0.7205 | 0.72 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs025_s1 | goal disturbed DR 1.75 | 0.3869 | 0.39 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs025_s1 | goal undisturbed DR 2.5 | 0.6591 | 0.66 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs025_s1 | goal disturbed DR 2.5 | 0.3244 | 0.32 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs035_s0 | goal undisturbed DR 1 | 0.7210 | 0.72 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs035_s0 | goal disturbed DR 1 | 0.4523 | 0.45 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs035_s0 | goal undisturbed DR 1.75 | 0.7074 | 0.71 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs035_s0 | goal disturbed DR 1.75 | 0.4040 | 0.4 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs035_s0 | goal undisturbed DR 2.5 | 0.6324 | 0.63 | 0.006 |
| S81-S83 | Table 11 g18_ret_rs035_s0 | goal disturbed DR 2.5 | 0.3153 | 0.32 | 0.006 |
| S84 | tables g18_ret_rs025_s2 | goal undisturbed DR 1 | 0.7580 | 0.76 | 0.006 |
| S84 | tables g18_ret_rs025_s2 | goal disturbed DR 1 | 0.4409 | 0.44 | 0.006 |
| S84 | tables g18_ret_rs025_s2 | goal undisturbed DR 1.75 | 0.7080 | 0.71 | 0.006 |
| S84 | tables g18_ret_rs025_s2 | goal disturbed DR 1.75 | 0.4131 | 0.41 | 0.006 |
| S84 | tables g18_ret_rs025_s2 | goal undisturbed DR 2.5 | 0.6295 | 0.63 | 0.006 |
| S84 | tables g18_ret_rs025_s2 | goal disturbed DR 2.5 | 0.3176 | 0.32 | 0.006 |
| S84 | tables g20_ret_rs025_depth_s0 | goal undisturbed DR 1 | 0.7386 | 0.74 | 0.006 |
| S84 | tables g20_ret_rs025_depth_s0 | goal disturbed DR 1 | 0.4312 | 0.43 | 0.006 |
| S84 | tables g20_ret_rs025_depth_s0 | goal undisturbed DR 1.75 | 0.6920 | 0.69 | 0.006 |
| S84 | tables g20_ret_rs025_depth_s0 | goal disturbed DR 1.75 | 0.3818 | 0.38 | 0.006 |
| S84 | tables g20_ret_rs025_depth_s0 | goal undisturbed DR 2.5 | 0.6267 | 0.63 | 0.006 |
| S84 | tables g20_ret_rs025_depth_s0 | goal disturbed DR 2.5 | 0.3068 | 0.31 | 0.006 |
| S84 | tables g19_ret_rs025_400m_s0 | goal undisturbed DR 1 | 0.7506 | 0.75 | 0.006 |
| S84 | tables g19_ret_rs025_400m_s0 | goal disturbed DR 1 | 0.4273 | 0.43 | 0.006 |
| S84 | tables g19_ret_rs025_400m_s0 | goal undisturbed DR 1.75 | 0.7045 | 0.7 | 0.006 |
| S84 | tables g19_ret_rs025_400m_s0 | goal disturbed DR 1.75 | 0.3602 | 0.36 | 0.006 |
| S84 | tables g19_ret_rs025_400m_s0 | goal undisturbed DR 2.5 | 0.6358 | 0.64 | 0.006 |
| S84 | tables g19_ret_rs025_400m_s0 | goal disturbed DR 2.5 | 0.2994 | 0.3 | 0.006 |
| S84 | text | 400M vel falls undisturbed | 0.3778 | 0.38 | 0.006 |
| S84 | text | depth vel falls undisturbed | 0.3545 | 0.35 | 0.006 |
| S83/S84 | Table 13 | flat seed 0 | 0.0795 | 0.08 | 0.006 |
| S83/S84 | Table 13 | flat seed 1 | 0.0739 | 0.07 | 0.006 |
| S84 | Table 13 | flat depth | 0.0795 | 0.08 | 0.006 |
| S83/S84 | Table 13 | step_up seed 0 | 0.6080 | 0.61 | 0.006 |
| S83/S84 | Table 13 | step_up seed 1 | 0.6250 | 0.62 | 0.006 |
| S84 | Table 13 | step_up depth | 0.5455 | 0.55 | 0.006 |
| S83/S84 | Table 13 | large_drop seed 0 | 0.6364 | 0.64 | 0.006 |
| S83/S84 | Table 13 | large_drop seed 1 | 0.6591 | 0.66 | 0.006 |
| S84 | Table 13 | large_drop depth | 0.6761 | 0.68 | 0.006 |
| S83/S84 | Table 13 | bump seed 0 | 0.7670 | 0.77 | 0.006 |
| S83/S84 | Table 13 | bump seed 1 | 0.7443 | 0.74 | 0.006 |
| S84 | Table 13 | bump depth | 0.7500 | 0.75 | 0.006 |
| S83/S84 | Table 13 | stairs seed 0 | 0.4830 | 0.48 | 0.006 |
| S83/S84 | Table 13 | stairs seed 1 | 0.5170 | 0.52 | 0.006 |
| S84 | Table 13 | stairs depth | 0.5057 | 0.51 | 0.006 |
| S83/S84 | Table 13 | ramp_up seed 0 | 0.1193 | 0.12 | 0.006 |
| S83/S84 | Table 13 | ramp_up seed 1 | 0.1364 | 0.14 | 0.006 |
| S83/S84 | Table 13 | ramp_down seed 0 | 0.1989 | 0.2 | 0.006 |
| S83/S84 | Table 13 | ramp_down seed 1 | 0.2614 | 0.26 | 0.006 |
| S83/S84 | Table 13 | rough seed 0 | 0.1705 | 0.17 | 0.006 |
| S83/S84 | Table 13 | rough seed 1 | 0.1648 | 0.16 | 0.006 |
| S83/S84 | Table 13 | step_down seed 0 | 0.1591 | 0.16 | 0.006 |
| S83/S84 | Table 13 | step_down seed 1 | 0.1648 | 0.16 | 0.006 |
| S83/S84 | Table 13 | small_drop seed 0 | 0.1761 | 0.18 | 0.006 |
| S83/S84 | Table 13 | small_drop seed 1 | 0.1705 | 0.17 | 0.006 |
| S85-R | Table 24 | g18_ret_rs025_s1 s1.5 undisturbed | 0.7125 | 0.71 | 0.006 |
| S85-R | Table 24 | g18_ret_rs025_s1 tau3 undisturbed | 0.3000 | 0.3 | 0.006 |
| S85-R | Table 24 | g18_ret_rs025_s1 s1.5 disturbed | 0.6562 | 0.66 | 0.006 |
| S85-R | Table 24 | g18_ret_rs025_s1 tau3 disturbed | 0.1500 | 0.15 | 0.006 |
| S85-R | Table 24 | g12_dr175_s1 s1.5 undisturbed | 0.6875 | 0.69 | 0.006 |
| S85-R | Table 24 | g12_dr175_s1 tau3 undisturbed | 0.1500 | 0.15 | 0.006 |
| S85-R | Table 24 | g12_dr175_s1 s1.5 disturbed | 0.6125 | 0.61 | 0.006 |
| S85-R | Table 24 | g12_dr175_s1 tau3 disturbed | 0.1250 | 0.12 | 0.006 |
| S85-R | Table 24 | g16_pure_ret100_s1 s1.5 undisturbed | 0.0875 | 0.09 | 0.006 |
| S85-R | Table 24 | g16_pure_ret100_s1 tau3 undisturbed | 0.0000 | 0.0 | 0.006 |
| S85-R | Table 24 | g16_pure_ret100_s1 s1.5 disturbed | 0.1500 | 0.15 | 0.006 |
| S85-R | Table 24 | g16_pure_ret100_s1 tau3 disturbed | 0.0000 | 0.0 | 0.006 |
| S85-R | Table 24 | g21_ret_rs025_tau3_s0 s1.5 undisturbed | 0.8938 | 0.89 | 0.006 |
| S85-R | Table 24 | g21_ret_rs025_tau3_s0 tau3 undisturbed | 0.2938 | 0.29 | 0.006 |
| S85-R | Table 24 | g21_ret_rs025_tau3_s0 s1.5 disturbed | 0.7625 | 0.76 | 0.006 |
| S85-R | Table 24 | g21_ret_rs025_tau3_s0 tau3 disturbed | 0.1750 | 0.17 | 0.006 |
| S85-R | Table 24 | g21_ret_rs025_tau3_s1 s1.5 undisturbed | 0.7500 | 0.75 | 0.006 |
| S85-R | Table 24 | g21_ret_rs025_tau3_s1 tau3 undisturbed | 0.2562 | 0.26 | 0.006 |
| S85-R | Table 24 | g21_ret_rs025_tau3_s1 s1.5 disturbed | 0.6562 | 0.66 | 0.006 |
| S85-R | Table 24 | g21_ret_rs025_tau3_s1 tau3 disturbed | 0.2125 | 0.21 | 0.006 |
| S87 | Table 14 | 0.1 m/s all | 0.4250 | 0.42 | 0.006 |
| S87 | Table 14 | 0.1 m/s step_up | 0.6364 | 0.64 | 0.006 |
| S87 | Table 14 | 0.1 m/s bump | 0.6420 | 0.64 | 0.006 |
| S87 | Table 14 | 0.1 m/s large_drop | 0.8011 | 0.8 | 0.006 |
| S87 | Table 14 | 0.1 m/s stairs | 0.5511 | 0.55 | 0.006 |
| S87 | Table 14 | 0.1 m/s flat | 0.1534 | 0.15 | 0.006 |
| S87 | Table 14 | 0.2 m/s all | 0.4068 | 0.41 | 0.006 |
| S87 | Table 14 | 0.2 m/s step_up | 0.6591 | 0.66 | 0.006 |
| S87 | Table 14 | 0.2 m/s bump | 0.8011 | 0.8 | 0.006 |
| S87 | Table 14 | 0.2 m/s large_drop | 0.7273 | 0.73 | 0.006 |
| S87 | Table 14 | 0.2 m/s stairs | 0.5625 | 0.56 | 0.006 |
| S87 | Table 14 | 0.2 m/s flat | 0.0795 | 0.08 | 0.006 |
| S87 | Table 14 | 0.3 m/s all | 0.3727 | 0.37 | 0.006 |
| S87 | Table 14 | 0.3 m/s step_up | 0.6420 | 0.64 | 0.006 |
| S87 | Table 14 | 0.3 m/s bump | 0.7386 | 0.74 | 0.006 |
| S87 | Table 14 | 0.3 m/s large_drop | 0.6648 | 0.66 | 0.006 |
| S87 | Table 14 | 0.3 m/s stairs | 0.5170 | 0.52 | 0.006 |
| S87 | Table 14 | 0.3 m/s flat | 0.0966 | 0.1 | 0.006 |
| S87 | Table 15 | step_up smaller half | 0.3960 | 0.4 | 0.006 |
| S87 | Table 15 | step_up larger half | 0.9467 | 0.95 | 0.006 |
| S87 | Table 15 | bump smaller half | 0.5852 | 0.59 | 0.006 |
| S87 | Table 15 | bump larger half | 0.9034 | 0.9 | 0.006 |
| S87 | Table 15 | stairs smaller half | 0.2727 | 0.27 | 0.006 |
| S87 | Table 15 | stairs larger half | 0.7045 | 0.7 | 0.006 |
| S87 | Table 15 | large_drop smaller half | 0.6136 | 0.61 | 0.006 |
| S87 | Table 15 | large_drop larger half | 0.7727 | 0.77 | 0.006 |
| S87 | Table 15 | small_drop smaller half | 0.2338 | 0.23 | 0.006 |
| S87 | Table 15 | small_drop larger half | 0.3377 | 0.34 | 0.006 |
| S87 | Table 15 | step_down smaller half | 0.1200 | 0.12 | 0.006 |
| S87 | Table 15 | step_down larger half | 0.2105 | 0.21 | 0.006 |
| S87 | text | governor off falls | 0.3949 | 0.39 | 0.006 |
| S87 | text | governor on falls | 0.3682 | 0.37 | 0.006 |
| S100 | Table 35 | 0.1 m/s all | 0.2801 | 0.28 | 0.006 |
| S100 | Table 35 | 0.1 m/s step_up | 0.4602 | 0.46 | 0.006 |
| S100 | Table 35 | 0.1 m/s bump | 0.5398 | 0.54 | 0.006 |
| S100 | Table 35 | 0.1 m/s large_drop | 0.7386 | 0.74 | 0.006 |
| S100 | Table 35 | 0.1 m/s stairs | 0.4318 | 0.43 | 0.006 |
| S100 | Table 35 | 0.1 m/s flat | 0.0568 | 0.06 | 0.006 |
| S100 | Table 35 | 0.2 m/s all | 0.3415 | 0.34 | 0.006 |
| S100 | Table 35 | 0.2 m/s step_up | 0.6080 | 0.61 | 0.006 |
| S100 | Table 35 | 0.2 m/s bump | 0.7273 | 0.73 | 0.006 |
| S100 | Table 35 | 0.2 m/s large_drop | 0.7216 | 0.72 | 0.006 |
| S100 | Table 35 | 0.2 m/s stairs | 0.4830 | 0.48 | 0.006 |
| S100 | Table 35 | 0.2 m/s flat | 0.0455 | 0.05 | 0.006 |
| S100 | Table 35 | 0.3 m/s all | 0.3489 | 0.35 | 0.006 |
| S100 | Table 35 | 0.3 m/s step_up | 0.6193 | 0.62 | 0.006 |
| S100 | Table 35 | 0.3 m/s bump | 0.7386 | 0.74 | 0.006 |
| S100 | Table 35 | 0.3 m/s large_drop | 0.6648 | 0.66 | 0.006 |
| S100 | Table 35 | 0.3 m/s stairs | 0.4886 | 0.49 | 0.006 |
| S100 | Table 35 | 0.3 m/s flat | 0.0625 | 0.06 | 0.006 |
| S100 | Table 35 | 0.6 m/s all | 0.4023 | 0.4 | 0.006 |
| S100 | Table 35 | 0.6 m/s step_up | 0.5398 | 0.54 | 0.006 |
| S100 | Table 35 | 0.6 m/s bump | 0.6648 | 0.66 | 0.006 |
| S100 | Table 35 | 0.6 m/s large_drop | 0.7670 | 0.77 | 0.006 |
| S100 | Table 35 | 0.6 m/s stairs | 0.5114 | 0.51 | 0.006 |
| S100 | Table 35 | 0.6 m/s flat | 0.1080 | 0.11 | 0.006 |
| S100 | Table 35 | 0.9 m/s all | 0.5062 | 0.51 | 0.006 |
| S100 | Table 35 | 0.9 m/s step_up | 0.6250 | 0.62 | 0.006 |
| S100 | Table 35 | 0.9 m/s bump | 0.6591 | 0.66 | 0.006 |
| S100 | Table 35 | 0.9 m/s large_drop | 0.8295 | 0.83 | 0.006 |
| S100 | Table 35 | 0.9 m/s stairs | 0.6250 | 0.62 | 0.006 |
| S100 | Table 35 | 0.9 m/s flat | 0.2273 | 0.23 | 0.006 |
| S103 addendum | text | speed-layer effect, smallest | 0.0300 | 0.03 | 0.006 |
| S103 addendum | text | speed-layer effect, largest | 0.0500 | 0.05 | 0.006 |
| S103 addendum | text | cost in tile exits, smallest | 0.0400 | 0.04 | 0.006 |
| S103 addendum | text | cost in tile exits, largest | 0.0800 | 0.08 | 0.006 |
| S103 addendum | Table 36 | 40 s none all | 0.3824 | 0.38 | 0.006 |
| S103 addendum | Table 36 | 40 s none step_up | 0.6250 | 0.62 | 0.006 |
| S103 addendum | Table 36 | 40 s none bump | 0.7898 | 0.79 | 0.006 |
| S103 addendum | Table 36 | 40 s none stairs | 0.5568 | 0.56 | 0.006 |
| S103 addendum | Table 36 | 40 s none large_drop | 0.7216 | 0.72 | 0.006 |
| S103 addendum | Table 36 | 40 s none flat | 0.1080 | 0.11 | 0.006 |
| S103 addendum | Table 36 | 40 s none exit | 0.6176 | 0.62 | 0.006 |
| S103 addendum | Table 36 | 40 s none prog | 1.3475 | 1.35 | 0.006 |
| S103 addendum | Table 36 | 40 s oracle all | 0.3261 | 0.33 | 0.006 |
| S103 addendum | Table 36 | 40 s oracle step_up | 0.6136 | 0.61 | 0.006 |
| S103 addendum | Table 36 | 40 s oracle bump | 0.7216 | 0.72 | 0.006 |
| S103 addendum | Table 36 | 40 s oracle stairs | 0.4318 | 0.43 | 0.006 |
| S103 addendum | Table 36 | 40 s oracle large_drop | 0.7443 | 0.74 | 0.006 |
| S103 addendum | Table 36 | 40 s oracle flat | 0.0739 | 0.07 | 0.006 |
| S103 addendum | Table 36 | 40 s oracle exit | 0.5818 | 0.58 | 0.006 |
| S103 addendum | Table 36 | 40 s oracle prog | 1.3137 | 1.31 | 0.006 |
| S103 addendum | Table 36 | 40 s tof all | 0.3455 | 0.35 | 0.006 |
| S103 addendum | Table 36 | 40 s tof step_up | 0.6136 | 0.61 | 0.006 |
| S103 addendum | Table 36 | 40 s tof bump | 0.7784 | 0.78 | 0.006 |
| S103 addendum | Table 36 | 40 s tof stairs | 0.4659 | 0.47 | 0.006 |
| S103 addendum | Table 36 | 40 s tof large_drop | 0.7670 | 0.77 | 0.006 |
| S103 addendum | Table 36 | 40 s tof flat | 0.0852 | 0.09 | 0.006 |
| S103 addendum | Table 36 | 40 s tof exit | 0.5403 | 0.54 | 0.006 |
| S103 addendum | Table 36 | 40 s tof prog | 1.2774 | 1.28 | 0.006 |
| S103 addendum | text | goal 30 s none falls | 0.2381 | 0.24 | 0.006 |
| S103 addendum | text | goal 30 s none success | 0.7619 | 0.76 | 0.006 |
| S103 addendum | text | goal 30 s tof05 falls | 0.2256 | 0.23 | 0.006 |
| S103 addendum | text | goal 30 s tof05 success | 0.6244 | 0.62 | 0.006 |
| S103 addendum | text | goal 30 s tof10 falls | 0.2528 | 0.25 | 0.006 |
| S103 addendum | text | goal 30 s tof10 success | 0.7415 | 0.74 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s0 none all | 0.3602 | 0.36 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s0 none step_up | 0.6193 | 0.62 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s0 none bump | 0.7386 | 0.74 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s0 none stairs | 0.4943 | 0.49 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s0 none large_drop | 0.7330 | 0.73 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s0 none flat | 0.0795 | 0.08 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s0 none prog | 1.3625 | 1.36 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s0 gov all | 0.1875 | 0.19 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s0 gov step_up | 0.3409 | 0.34 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s0 gov bump | 0.1534 | 0.15 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s0 gov stairs | 0.1818 | 0.18 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s0 gov large_drop | 0.7216 | 0.72 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s0 gov flat | 0.0682 | 0.07 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s0 gov prog | 1.0646 | 1.06 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s1 none all | 0.3625 | 0.36 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s1 none step_up | 0.6420 | 0.64 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s1 none bump | 0.7330 | 0.73 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s1 none stairs | 0.4830 | 0.48 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s1 none large_drop | 0.7273 | 0.73 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s1 none flat | 0.0511 | 0.05 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s1 none prog | 1.3491 | 1.35 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s1 gov all | 0.1795 | 0.18 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s1 gov step_up | 0.2841 | 0.28 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s1 gov bump | 0.1761 | 0.18 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s1 gov stairs | 0.1477 | 0.15 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s1 gov large_drop | 0.6875 | 0.69 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s1 gov flat | 0.0625 | 0.06 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s1 gov prog | 1.0677 | 1.07 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s2 none all | 0.3795 | 0.38 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s2 none step_up | 0.6250 | 0.62 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s2 none bump | 0.7443 | 0.74 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s2 none stairs | 0.5341 | 0.53 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s2 none large_drop | 0.7386 | 0.74 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s2 none flat | 0.1023 | 0.1 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s2 none prog | 1.3467 | 1.35 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s2 gov all | 0.1835 | 0.18 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s2 gov step_up | 0.3068 | 0.31 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s2 gov bump | 0.1364 | 0.14 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s2 gov stairs | 0.1023 | 0.1 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s2 gov large_drop | 0.7386 | 0.74 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s2 gov flat | 0.0739 | 0.07 | 0.006 |
| S101 (g18 rows, stand per S102) | Table 36 | g18_ret_rs025_s2 gov prog | 1.0284 | 1.03 | 0.006 |
| S101 (g18 rows) | text | g18 progress loss | 0.2212 | 0.22 | 0.006 |
| S102 | Table 36 | g18 s1 matched pair, no governor | 0.3483 | 0.35 | 0.006 |
| S102 | Table 36 | g18 s1 matched pair, governor | 0.1744 | 0.17 | 0.006 |
| S102 | Table 36 | g26 s1 matched pair, no governor | 0.3744 | 0.37 | 0.006 |
| S102 | Table 36 | g26 s1 matched pair, governor | 0.1574 | 0.16 | 0.006 |
| S100 addendum | text | g18 s1 governor slow to 0.1 m/s falls | 0.2614 | 0.26 | 0.006 |
| S88 | text | traction limit 9 cm ball [cm] | 2.6360 | 2.6 | 0.06 |
| S88 | text | traction limit 13.5 cm ball [cm] | 3.9541 | 4.0 | 0.06 |
| S88 | Table 16 | step_up nominal smaller | 0.3973 | 0.4 | 0.006 |
| S88 | Table 16 | step_up nominal larger | 0.9818 | 0.98 | 0.006 |
| S88 | Table 16 | step_up s1.5 smaller | 0.2603 | 0.26 | 0.006 |
| S88 | Table 16 | step_up s1.5 larger | 0.8000 | 0.8 | 0.006 |
| S88 | Table 16 | bump nominal smaller | 0.5156 | 0.52 | 0.006 |
| S88 | Table 16 | bump nominal larger | 0.9375 | 0.94 | 0.006 |
| S88 | Table 16 | bump s1.5 smaller | 0.3281 | 0.33 | 0.006 |
| S88 | Table 16 | bump s1.5 larger | 0.6250 | 0.62 | 0.006 |
| S88 | Table 16 | stairs nominal smaller | 0.1719 | 0.17 | 0.006 |
| S88 | Table 16 | stairs nominal larger | 0.6719 | 0.67 | 0.006 |
| S88 | Table 16 | stairs s1.5 smaller | 0.1875 | 0.19 | 0.006 |
| S88 | Table 16 | stairs s1.5 larger | 0.6406 | 0.64 | 0.006 |
| S88 | Table 16 | large_drop nominal smaller | 0.6406 | 0.64 | 0.006 |
| S88 | Table 16 | large_drop nominal larger | 0.8281 | 0.83 | 0.006 |
| S88 | Table 16 | large_drop s1.5 smaller | 0.6250 | 0.62 | 0.006 |
| S88 | Table 16 | large_drop s1.5 larger | 0.7188 | 0.72 | 0.006 |
| S88 | Table 17 | g22_ret_rs025_scan_s0 goal undisturbed | 0.6841 | 0.68 | 0.006 |
| S88 | Table 17 | g22_ret_rs025_scan_s0 goal disturbed | 0.4136 | 0.41 | 0.006 |
| S88 | Table 17 | g22_ret_rs025_scan_s0 vel falls | 0.3608 | 0.36 | 0.006 |
| S88 | Table 17 | g22_ret_rs025_scan_s0 step_up | 0.5909 | 0.59 | 0.006 |
| S88 | Table 17 | g22_ret_rs025_scan_s0 bump | 0.7841 | 0.78 | 0.006 |
| S88 | Table 17 | g22_ret_rs025_scan_s0 stairs | 0.5114 | 0.51 | 0.006 |
| S88 | Table 17 | g22_ret_rs025_scan_s0 large_drop | 0.7045 | 0.7 | 0.006 |
| S88 | Table 17 | g22_ret_rs05_scan_s0 goal undisturbed | 0.7375 | 0.74 | 0.006 |
| S88 | Table 17 | g22_ret_rs05_scan_s0 goal disturbed | 0.5324 | 0.53 | 0.006 |
| S88 | Table 17 | g22_ret_rs05_scan_s0 vel falls | 0.3210 | 0.32 | 0.006 |
| S88 | Table 17 | g22_ret_rs05_scan_s0 step_up | 0.5455 | 0.55 | 0.006 |
| S88 | Table 17 | g22_ret_rs05_scan_s0 bump | 0.7102 | 0.71 | 0.006 |
| S88 | Table 17 | g22_ret_rs05_scan_s0 stairs | 0.4545 | 0.45 | 0.006 |
| S88 | Table 17 | g22_ret_rs05_scan_s0 large_drop | 0.6307 | 0.63 | 0.006 |
| S88 | Table 17 | g22_pure_ret_scan_s0 goal undisturbed | 0.6915 | 0.69 | 0.006 |
| S88 | Table 17 | g22_pure_ret_scan_s0 goal disturbed | 0.4972 | 0.5 | 0.006 |
| S88 | Table 17 | g22_pure_ret_scan_s0 step_up | 0.2898 | 0.29 | 0.006 |
| S88 | Table 17 | g22_pure_ret_scan_s0 bump | 0.2386 | 0.24 | 0.006 |
| S88 | Table 17 | g22_pure_ret_scan_s0 large_drop | 0.3807 | 0.38 | 0.006 |
| S88 | Table 18 | step_up pure falls | 0.6909 | 0.69 | 0.006 |
| S88 | Table 18 | step_up pure crossed | 0.0182 | 0.02 | 0.006 |
| S88 | Table 18 | step_up lqr falls | 0.9818 | 0.98 | 0.006 |
| S88 | Table 18 | step_up lqr crossed | 0.0182 | 0.02 | 0.006 |
| S88 | Table 18 | bump pure falls | 0.2969 | 0.3 | 0.006 |
| S88 | Table 18 | bump pure crossed | 0.0156 | 0.02 | 0.006 |
| S88 | Table 18 | bump lqr falls | 0.9375 | 0.94 | 0.006 |
| S88 | Table 18 | bump lqr crossed | 0.0625 | 0.06 | 0.006 |
| S88 | Table 18 | stairs pure falls | 0.3750 | 0.38 | 0.006 |
| S88 | Table 18 | stairs pure crossed | 0.0312 | 0.03 | 0.006 |
| S88 | Table 18 | stairs lqr falls | 0.6719 | 0.67 | 0.006 |
| S88 | Table 18 | stairs lqr crossed | 0.3281 | 0.33 | 0.006 |
| S88 | Table 18 | large_drop pure falls | 0.5781 | 0.58 | 0.006 |
| S88 | Table 18 | large_drop pure crossed | 0.3906 | 0.39 | 0.006 |
| S88 | Table 18 | large_drop lqr falls | 0.8281 | 0.83 | 0.006 |
| S88 | Table 18 | large_drop lqr crossed | 0.1719 | 0.17 | 0.006 |
| S89 | Tables 19/20 | nav_blind fall | 0.9129 | 0.91 | 0.006 |
| S89 | Tables 19/20 | nav_blind success | 0.0871 | 0.09 | 0.006 |
| S89 | Tables 19/20 | nav_blind stuck | 0.0000 | 0.0 | 0.006 |
| S89 | Tables 19/20 | nav_blind over_block | 0.7443 | 0.74 | 0.006 |
| S89 | Tables 19/20 | nav_oracle fall | 0.0890 | 0.09 | 0.006 |
| S89 | Tables 19/20 | nav_oracle success | 0.9072 | 0.91 | 0.006 |
| S89 | Tables 19/20 | nav_oracle stuck | 0.0038 | 0.0 | 0.006 |
| S89 | Tables 19/20 | nav_oracle over_block | 0.0038 | 0.0 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_head fall | 0.1080 | 0.11 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_head success | 0.8523 | 0.85 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_head stuck | 0.0398 | 0.04 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_head over_block | 0.0208 | 0.02 | 0.006 |
| S89 | Tables 19/20 | nav_cam32_head fall | 0.1250 | 0.13 | 0.006 |
| S89 | Tables 19/20 | nav_cam32_head success | 0.8277 | 0.83 | 0.006 |
| S89 | Tables 19/20 | nav_cam32_head stuck | 0.0473 | 0.05 | 0.006 |
| S89 | Tables 19/20 | nav_cam32_head over_block | 0.0246 | 0.02 | 0.006 |
| S89 | Tables 19/20 | nav_cam32_head_p40 fall | 0.1515 | 0.15 | 0.006 |
| S89 | Tables 19/20 | nav_cam32_head_p40 success | 0.5114 | 0.51 | 0.006 |
| S89 | Tables 19/20 | nav_cam32_head_p40 stuck | 0.3371 | 0.34 | 0.006 |
| S89 | Tables 19/20 | nav_cam32_head_p40 over_block | 0.0019 | 0.0 | 0.006 |
| S89 | Tables 19/20 | nav_tof8_head fall | 0.1402 | 0.14 | 0.006 |
| S89 | Tables 19/20 | nav_tof8_head success | 0.7330 | 0.73 | 0.006 |
| S89 | Tables 19/20 | nav_tof8_head stuck | 0.1269 | 0.13 | 0.006 |
| S89 | Tables 19/20 | nav_tof8_head over_block | 0.0227 | 0.02 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_low fall | 0.1080 | 0.11 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_low success | 0.7841 | 0.78 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_low stuck | 0.1080 | 0.11 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_low over_block | 0.0152 | 0.02 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_cones fall | 0.3523 | 0.35 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_cones success | 0.5814 | 0.58 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_cones stuck | 0.0663 | 0.07 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_cones over_block | 0.2519 | 0.25 | 0.006 |
| S89 | Tables 19/20 | nav_blind_pp fall | 0.8580 | 0.86 | 0.006 |
| S89 | Tables 19/20 | nav_blind_pp success | 0.1420 | 0.14 | 0.006 |
| S89 | Tables 19/20 | nav_blind_pp stuck | 0.0000 | 0.0 | 0.006 |
| S89 | Tables 19/20 | nav_blind_pp over_block | 0.4432 | 0.44 | 0.006 |
| S89 | Tables 19/20 | nav_oracle_pp fall | 0.4148 | 0.41 | 0.006 |
| S89 | Tables 19/20 | nav_oracle_pp success | 0.5777 | 0.58 | 0.006 |
| S89 | Tables 19/20 | nav_oracle_pp stuck | 0.0076 | 0.01 | 0.006 |
| S89 | Tables 19/20 | nav_oracle_pp over_block | 0.1326 | 0.13 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_head_pp fall | 0.5246 | 0.52 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_head_pp success | 0.4451 | 0.45 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_head_pp stuck | 0.0303 | 0.03 | 0.006 |
| S89 | Tables 19/20 | nav_tof18_head_pp over_block | 0.1458 | 0.15 | 0.006 |
| S89 | Table 20 | nav_tof18_hw0.30 success | 0.1061 | 0.11 | 0.006 |
| S89 | Table 20 | nav_tof18_hw0.30 stuck | 0.7027 | 0.7 | 0.006 |
| S89 | Table 20 | nav_tof18_hw0.38 success | 0.0019 | 0.0 | 0.006 |
| S89 | Table 20 | nav_tof18_hw0.38 stuck | 0.7140 | 0.71 | 0.006 |
| S89 | Table 20 | oracle pp 0.38 success | 0.0019 | 0.0 | 0.006 |
| S89 | Table 20 | oracle pp 0.38 falls | 0.5739 | 0.57 | 0.006 |
| S90/S91 | Tables 21/22 | navS_oracle success | 0.8580 | 0.86 | 0.006 |
| S90/S91 | Tables 21/22 | navS_oracle fall | 0.1383 | 0.14 | 0.006 |
| S90/S91 | Tables 21/22 | navS_oracle stuck | 0.0038 | 0.0 | 0.006 |
| S90/S91 | Tables 21/22 | navS_oracle over_block | 0.0398 | 0.04 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18 success | 0.7235 | 0.72 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18 fall | 0.2235 | 0.22 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18 stuck | 0.0530 | 0.05 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18 over_block | 0.1629 | 0.16 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18_aim_w2.0 success | 0.7841 | 0.78 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18_aim_w2.0 fall | 0.1629 | 0.16 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18_aim_w2.0 stuck | 0.0530 | 0.05 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18_aim_w2.0 over_block | 0.0265 | 0.03 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18_aim_w4.0 success | 0.7614 | 0.76 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18_aim_w4.0 fall | 0.1705 | 0.17 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18_aim_w4.0 stuck | 0.0682 | 0.07 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18_aim_w4.0 over_block | 0.0379 | 0.04 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18_w4 success | 0.7424 | 0.74 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18_rp4 success | 0.6856 | 0.69 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18_prox0.3 success | 0.4413 | 0.44 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18_prox0.3 stuck | 0.2784 | 0.28 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18_prox0.5 success | 0.2557 | 0.26 | 0.006 |
| S90/S91 | Tables 21/22 | navS_tof18_prox0.5 stuck | 0.4583 | 0.46 | 0.006 |
| S90 | Table 21 | smooth heading-rate RMS | 1.7299 | 1.7 | 0.06 |
| S90 | Table 22 | aim 4 rad/s heading-rate RMS | 3.2893 | 3.3 | 0.06 |
| S91 | Table 23 | g18_ret_rs025_s1 navS_oracle success | 0.8580 | 0.86 | 0.006 |
| S91 | Table 23 | g18_ret_rs025_s1 navS_oracle fall | 0.1383 | 0.14 | 0.006 |
| S91 | Table 23 | g18_ret_rs025_s1 navS_oracle on block | 0.0398 | 0.04 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s0 navS_oracle success | 0.8523 | 0.85 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s0 navS_oracle fall | 0.1288 | 0.13 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s0 navS_oracle on block | 0.0114 | 0.01 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s1 navS_oracle success | 0.8542 | 0.85 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s1 navS_oracle fall | 0.1402 | 0.14 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s1 navS_oracle on block | 0.0227 | 0.02 | 0.006 |
| S91 | Table 23 | g18_ret_rs025_s1 navS_oracle_pp success | 0.5455 | 0.55 | 0.006 |
| S91 | Table 23 | g18_ret_rs025_s1 navS_oracle_pp fall | 0.4432 | 0.44 | 0.006 |
| S91 | Table 23 | g18_ret_rs025_s1 navS_oracle_pp on block | 0.1989 | 0.2 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s0 navS_oracle_pp success | 0.5511 | 0.55 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s0 navS_oracle_pp fall | 0.4261 | 0.43 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s0 navS_oracle_pp on block | 0.1307 | 0.13 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s1 navS_oracle_pp success | 0.5682 | 0.57 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s1 navS_oracle_pp fall | 0.4186 | 0.42 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s1 navS_oracle_pp on block | 0.1269 | 0.13 | 0.006 |
| S91 | Table 23 | g18_ret_rs025_s1 navS_tof18 success | 0.7235 | 0.72 | 0.006 |
| S91 | Table 23 | g18_ret_rs025_s1 navS_tof18 fall | 0.2235 | 0.22 | 0.006 |
| S91 | Table 23 | g18_ret_rs025_s1 navS_tof18 on block | 0.1629 | 0.16 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s0 navS_tof18 success | 0.6818 | 0.68 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s0 navS_tof18 fall | 0.2462 | 0.25 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s0 navS_tof18 on block | 0.1515 | 0.15 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s1 navS_tof18 success | 0.7102 | 0.71 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s1 navS_tof18 fall | 0.2330 | 0.23 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s1 navS_tof18 on block | 0.1686 | 0.17 | 0.006 |
| S91 | Table 23 | g18_ret_rs025_s1 navS_tof18_pp success | 0.4053 | 0.41 | 0.006 |
| S91 | Table 23 | g18_ret_rs025_s1 navS_tof18_pp fall | 0.5606 | 0.56 | 0.006 |
| S91 | Table 23 | g18_ret_rs025_s1 navS_tof18_pp on block | 0.2216 | 0.22 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s0 navS_tof18_pp success | 0.4337 | 0.43 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s0 navS_tof18_pp fall | 0.5265 | 0.53 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s0 navS_tof18_pp on block | 0.1799 | 0.18 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s1 navS_tof18_pp success | 0.4564 | 0.46 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s1 navS_tof18_pp fall | 0.5133 | 0.51 | 0.006 |
| S91 | Table 23 | g23_nav_rs025_s1 navS_tof18_pp on block | 0.2064 | 0.21 | 0.006 |
| S104 | text | ToF within ... of the perfect map | 0.0500 | 0.05 | 0.006 |
| S104 | Table 37 | blind success | 0.0890 | 0.09 | 0.006 |
| S104 | Table 37 | blind fall | 0.9091 | 0.91 | 0.006 |
| S104 | Table 37 | blind stuck | 0.0019 | 0.0 | 0.006 |
| S104 | Table 37 | blind over_block | 0.7462 | 0.75 | 0.006 |
| S104 | Table 37 | oracle success | 0.8864 | 0.89 | 0.006 |
| S104 | Table 37 | oracle fall | 0.0720 | 0.07 | 0.006 |
| S104 | Table 37 | oracle stuck | 0.0417 | 0.04 | 0.006 |
| S104 | Table 37 | oracle over_block | 0.0076 | 0.01 | 0.006 |
| S104 | Table 37 | tof success | 0.8447 | 0.84 | 0.006 |
| S104 | Table 37 | tof fall | 0.0720 | 0.07 | 0.006 |
| S104 | Table 37 | tof stuck | 0.0833 | 0.08 | 0.006 |
| S104 | Table 37 | tof over_block | 0.0360 | 0.04 | 0.006 |
| S104 | Table 37 | oracle_pp success | 0.5682 | 0.57 | 0.006 |
| S104 | Table 37 | oracle_pp fall | 0.3883 | 0.39 | 0.006 |
| S104 | Table 37 | oracle_pp stuck | 0.0436 | 0.04 | 0.006 |
| S104 | Table 37 | oracle_pp over_block | 0.1648 | 0.16 | 0.006 |
| S104 | Table 37 | tof_pp success | 0.5114 | 0.51 | 0.006 |
| S104 | Table 37 | tof_pp fall | 0.4318 | 0.43 | 0.006 |
| S104 | Table 37 | tof_pp stuck | 0.0568 | 0.06 | 0.006 |
| S104 | Table 37 | tof_pp over_block | 0.1705 | 0.17 | 0.006 |
| S104 | Table 37 | oracle median time | 9.4800 | 9.5 | 0.06 |
| S104 | Table 37 | oracle heading-rate RMS | 1.6926 | 1.69 | 0.006 |
| S104 | Table 37 | tof median time | 10.7500 | 10.8 | 0.06 |
| S104 | Table 37 | tof heading-rate RMS | 1.7386 | 1.74 | 0.006 |
| S113 | index.html keynum | g34 flat achieved / command, largest (log: 1.06 at 0.9 m/s) | 1.0637 | 1.06 | 0.006 |
| S113 | index.html keynum | g34 flat achieved / command, smallest (0.96 on the page) | 0.9550 | 0.96 | 0.006 |
| S110 | Table 38 | g26_ret_rs025_mu01_s1 goal test fall | 0.1458 | 0.15 | 0.006 |
| S110 | Table 38 | g26_ret_rs025_mu01_s1 goal test rev | 1.1869 | 1.19 | 0.006 |
| S110 | Table 38 | g26_ret_rs025_mu01_s1 goal test hold | 0.0770 | 0.077 | 0.0006 |
| S113 | Table 38 | g32_mref_rsad_s0 goal test fall | 0.0833 | 0.083 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s0 goal test rev | 4.9385 | 4.94 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s0 goal test hold | 0.0959 | 0.096 | 0.0006 |
| S113 | Table 38 | g32_mref_rsad_s0 battery goal 1.u | 0.7608 | 0.76 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s0 battery goal 1.d | 0.4983 | 0.5 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s0 battery goal 1.75.u | 0.7256 | 0.73 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s0 battery goal 1.75.d | 0.4403 | 0.44 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s0 battery goal 2.5.u | 0.6767 | 0.68 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s0 battery goal 2.5.d | 0.3937 | 0.39 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s0 velocity falls DR1 undisturbed | 0.3193 | 0.32 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s0 velocity falls DR1 pushes | 0.4847 | 0.48 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s1 goal test fall | 0.1042 | 0.104 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s1 goal test rev | 0.5700 | 0.57 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s1 goal test hold | 0.0629 | 0.063 | 0.0006 |
| S113 | Table 38 | g32_mref_rsad_s1 battery goal 1.u | 0.7466 | 0.75 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s1 battery goal 1.d | 0.5193 | 0.52 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s1 battery goal 1.75.u | 0.7205 | 0.72 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s1 battery goal 1.75.d | 0.4614 | 0.46 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s1 battery goal 2.5.u | 0.6659 | 0.67 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s1 battery goal 2.5.d | 0.3830 | 0.38 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s1 velocity falls DR1 undisturbed | 0.3170 | 0.32 | 0.006 |
| S113 | Table 38 | g32_mref_rsad_s1 velocity falls DR1 pushes | 0.4784 | 0.48 | 0.006 |
| S113 | Table 38 | g33_mref_rsad_trk3_s0 goal test fall | 0.1004 | 0.1 | 0.006 |
| S113 | Table 38 | g33_mref_rsad_trk3_s0 goal test rev | 0.9894 | 0.99 | 0.006 |
| S113 | Table 38 | g33_mref_rsad_trk3_s0 goal test hold | 0.0683 | 0.068 | 0.0006 |
| S113 | Table 38 | g33_mref_rsad_trk3_s0 battery goal 1.u | 0.7727 | 0.77 | 0.006 |
| S113 | Table 38 | g33_mref_rsad_trk3_s0 battery goal 1.d | 0.5295 | 0.53 | 0.006 |
| S113 | Table 38 | g33_mref_rsad_trk3_s0 battery goal 1.75.u | 0.7460 | 0.75 | 0.006 |
| S113 | Table 38 | g33_mref_rsad_trk3_s0 battery goal 1.75.d | 0.4517 | 0.45 | 0.006 |
| S113 | Table 38 | g33_mref_rsad_trk3_s0 battery goal 2.5.u | 0.7011 | 0.7 | 0.006 |
| S113 | Table 38 | g33_mref_rsad_trk3_s0 battery goal 2.5.d | 0.4057 | 0.41 | 0.006 |
| S113 | Table 38 | g33_mref_rsad_trk3_s0 velocity falls DR1 undisturbed | 0.3170 | 0.32 | 0.006 |
| S113 | Table 38 | g33_mref_rsad_trk3_s0 velocity falls DR1 pushes | 0.4801 | 0.48 | 0.006 |
| S113 addendum | Table 38 | g33_mref_rsad_trk3_s1 goal test fall | 0.0814 | 0.081 | 0.006 |
| S113 addendum | Table 38 | g33_mref_rsad_trk3_s1 goal test rev | 1.3756 | 1.38 | 0.006 |
| S113 addendum | Table 38 | g33_mref_rsad_trk3_s1 goal test hold | 0.0603 | 0.06 | 0.0006 |
| S113 addendum | Table 38 | g33_mref_rsad_trk3_s1 battery goal 1.u | 0.7665 | 0.77 | 0.006 |
| S113 addendum | Table 38 | g33_mref_rsad_trk3_s1 battery goal 1.d | 0.5062 | 0.51 | 0.006 |
| S113 addendum | Table 38 | g33_mref_rsad_trk3_s1 battery goal 1.75.u | 0.7324 | 0.73 | 0.006 |
| S113 addendum | Table 38 | g33_mref_rsad_trk3_s1 battery goal 1.75.d | 0.4466 | 0.45 | 0.006 |
| S113 addendum | Table 38 | g33_mref_rsad_trk3_s1 battery goal 2.5.u | 0.6710 | 0.67 | 0.006 |
| S113 addendum | Table 38 | g33_mref_rsad_trk3_s1 battery goal 2.5.d | 0.3847 | 0.38 | 0.006 |
| S113 addendum | Table 38 | g33_mref_rsad_trk3_s1 velocity falls DR1 undisturbed | 0.3023 | 0.3 | 0.006 |
| S113 addendum | Table 38 | g33_mref_rsad_trk3_s1 velocity falls DR1 pushes | 0.4801 | 0.48 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s0 goal test fall | 0.1042 | 0.104 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s0 goal test rev | 0.6686 | 0.67 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s0 goal test hold | 0.0580 | 0.058 | 0.0006 |
| S113 | Table 38 | g34_mref_rsad_db_s0 battery goal 1.u | 0.7932 | 0.79 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s0 battery goal 1.d | 0.5091 | 0.51 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s0 battery goal 1.75.u | 0.7699 | 0.77 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s0 battery goal 1.75.d | 0.4699 | 0.47 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s0 battery goal 2.5.u | 0.6903 | 0.69 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s0 battery goal 2.5.d | 0.3864 | 0.39 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s0 velocity falls DR1 undisturbed | 0.3250 | 0.32 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s0 velocity falls DR1 pushes | 0.4778 | 0.48 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s1 goal test fall | 0.0909 | 0.091 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s1 goal test rev | 0.8583 | 0.86 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s1 goal test hold | 0.0654 | 0.065 | 0.0006 |
| S113 | Table 38 | g34_mref_rsad_db_s1 battery goal 1.u | 0.7682 | 0.77 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s1 battery goal 1.d | 0.5068 | 0.51 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s1 battery goal 1.75.u | 0.7278 | 0.73 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s1 battery goal 1.75.d | 0.4500 | 0.45 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s1 battery goal 2.5.u | 0.6795 | 0.68 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s1 battery goal 2.5.d | 0.3909 | 0.39 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s1 velocity falls DR1 undisturbed | 0.3199 | 0.32 | 0.006 |
| S113 | Table 38 | g34_mref_rsad_db_s1 velocity falls DR1 pushes | 0.4761 | 0.48 | 0.006 |
| S113 addendum | Table 38 | g34_mref_rsad_db_s2 goal test fall | 0.1098 | 0.11 | 0.006 |
| S113 addendum | Table 38 | g34_mref_rsad_db_s2 goal test rev | 0.4767 | 0.48 | 0.006 |
| S113 addendum | Table 38 | g34_mref_rsad_db_s2 goal test hold | 0.0520 | 0.052 | 0.0006 |
| S113 addendum | Table 38 | g34_mref_rsad_db_s2 battery goal 1.u | 0.7784 | 0.78 | 0.006 |
| S113 addendum | Table 38 | g34_mref_rsad_db_s2 battery goal 1.d | 0.5250 | 0.52 | 0.006 |
| S113 addendum | Table 38 | g34_mref_rsad_db_s2 battery goal 1.75.u | 0.7483 | 0.75 | 0.006 |
| S113 addendum | Table 38 | g34_mref_rsad_db_s2 battery goal 1.75.d | 0.4455 | 0.45 | 0.006 |
| S113 addendum | Table 38 | g34_mref_rsad_db_s2 battery goal 2.5.u | 0.6915 | 0.69 | 0.006 |
| S113 addendum | Table 38 | g34_mref_rsad_db_s2 battery goal 2.5.d | 0.3869 | 0.39 | 0.006 |
| S113 addendum | Table 38 | g34_mref_rsad_db_s2 velocity falls DR1 undisturbed | 0.3097 | 0.31 | 0.006 |
| S113 addendum | Table 38 | g34_mref_rsad_db_s2 velocity falls DR1 pushes | 0.4812 | 0.48 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s0 goal test fall | 0.0909 | 0.091 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s0 battery goal 1.u | 0.8062 | 0.81 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s0 battery goal 1.d | 0.5267 | 0.53 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s0 battery goal 1.75.u | 0.7790 | 0.78 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s0 battery goal 1.75.d | 0.4500 | 0.45 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s0 battery goal 2.5.u | 0.7244 | 0.72 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s0 battery goal 2.5.d | 0.3943 | 0.39 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s1 goal test fall | 0.1080 | 0.108 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s1 battery goal 1.u | 0.7858 | 0.79 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s1 battery goal 1.d | 0.5250 | 0.52 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s1 battery goal 1.75.u | 0.7722 | 0.77 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s1 battery goal 1.75.d | 0.4489 | 0.45 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s1 battery goal 2.5.u | 0.7176 | 0.72 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s1 battery goal 2.5.d | 0.3881 | 0.39 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s2 goal test fall | 0.0814 | 0.081 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s2 goal test rev | 0.6926 | 0.69 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s2 goal test hold | 0.0437 | 0.044 | 0.0006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s2 battery goal 1.u | 0.7699 | 0.77 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s2 battery goal 1.d | 0.5222 | 0.52 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s2 battery goal 1.75.u | 0.7273 | 0.73 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s2 battery goal 1.75.d | 0.4892 | 0.49 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s2 battery goal 2.5.u | 0.6670 | 0.67 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s2 battery goal 2.5.d | 0.3955 | 0.4 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s2 velocity falls DR1 undisturbed | 0.3051 | 0.31 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s2 velocity falls DR1 pushes | 0.4608 | 0.46 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s0 goal test fall | 0.0682 | 0.068 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s0 goal test rev | 3.7296 | 3.73 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s0 goal test hold | 0.0763 | 0.076 | 0.0006 |
| S115 | Table 38 | g36_deploy_400m_s0 battery goal 1.u | 0.7403 | 0.74 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s0 battery goal 1.d | 0.4670 | 0.47 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s0 battery goal 1.75.u | 0.7085 | 0.71 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s0 battery goal 1.75.d | 0.4159 | 0.42 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s0 battery goal 2.5.u | 0.6506 | 0.65 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s0 battery goal 2.5.d | 0.3540 | 0.35 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s0 velocity falls DR1 undisturbed | 0.3205 | 0.32 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s0 velocity falls DR1 pushes | 0.4903 | 0.49 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s1 goal test fall | 0.0720 | 0.072 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s1 goal test rev | 2.5235 | 2.52 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s1 goal test hold | 0.0671 | 0.067 | 0.0006 |
| S115 | Table 38 | g36_deploy_400m_s1 battery goal 1.u | 0.7335 | 0.73 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s1 battery goal 1.d | 0.4801 | 0.48 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s1 battery goal 1.75.u | 0.6903 | 0.69 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s1 battery goal 1.75.d | 0.4597 | 0.46 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s1 battery goal 2.5.u | 0.6523 | 0.65 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s1 battery goal 2.5.d | 0.3790 | 0.38 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s1 velocity falls DR1 undisturbed | 0.3153 | 0.32 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s1 velocity falls DR1 pushes | 0.4920 | 0.49 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s2 goal test fall | 0.0644 | 0.064 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s2 goal test rev | 3.3583 | 3.36 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s2 goal test hold | 0.0731 | 0.073 | 0.0006 |
| S115 | Table 38 | g36_deploy_400m_s2 battery goal 1.u | 0.7165 | 0.72 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s2 battery goal 1.d | 0.4540 | 0.45 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s2 battery goal 1.75.u | 0.6881 | 0.69 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s2 battery goal 1.75.d | 0.4136 | 0.41 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s2 battery goal 2.5.u | 0.6443 | 0.64 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s2 battery goal 2.5.d | 0.3409 | 0.34 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s2 velocity falls DR1 undisturbed | 0.3222 | 0.32 | 0.006 |
| S115 | Table 38 | g36_deploy_400m_s2 velocity falls DR1 pushes | 0.4864 | 0.49 | 0.006 |
| S116 | Table 38 | g37_push15_s0 goal test fall | 0.1364 | 0.136 | 0.006 |
| S116 | Table 38 | g37_push15_s0 battery goal 1.u | 0.7903 | 0.79 | 0.006 |
| S116 | Table 38 | g37_push15_s0 battery goal 1.d | 0.4886 | 0.49 | 0.006 |
| S116 | Table 38 | g37_push15_s0 battery goal 1.75.u | 0.7580 | 0.76 | 0.006 |
| S116 | Table 38 | g37_push15_s0 battery goal 1.75.d | 0.4341 | 0.43 | 0.006 |
| S116 | Table 38 | g37_push15_s0 battery goal 2.5.u | 0.7045 | 0.7 | 0.006 |
| S116 | Table 38 | g37_push15_s0 battery goal 2.5.d | 0.3619 | 0.36 | 0.006 |
| S116 | Table 38 | g37_push20_s0 goal test fall | 0.1193 | 0.119 | 0.006 |
| S116 | Table 38 | g37_push20_s0 battery goal 1.u | 0.7864 | 0.79 | 0.006 |
| S116 | Table 38 | g37_push20_s0 battery goal 1.d | 0.4898 | 0.49 | 0.006 |
| S116 | Table 38 | g37_push20_s0 battery goal 1.75.u | 0.7591 | 0.76 | 0.006 |
| S116 | Table 38 | g37_push20_s0 battery goal 1.75.d | 0.4540 | 0.45 | 0.006 |
| S116 | Table 38 | g37_push20_s0 battery goal 2.5.u | 0.7159 | 0.72 | 0.006 |
| S116 | Table 38 | g37_push20_s0 battery goal 2.5.d | 0.3648 | 0.36 | 0.006 |
| S115, S116 | Table 38 | g34_mref_rsad_db_s0 hard-design falls | 0.3194 | 0.32 | 0.006 |
| S115, S116 | Table 38 | g37_push15_s0 hard-design falls | 0.3681 | 0.37 | 0.006 |
| S115, S116 | Table 38 | g37_push20_s0 hard-design falls | 0.3681 | 0.37 | 0.006 |
| S115, S116 | Table 38 | g36_deploy_400m_s0 hard-design falls | 0.2292 | 0.23 | 0.006 |
| S115, S116 | Table 38 | g36_deploy_400m_s1 hard-design falls | 0.2500 | 0.25 | 0.006 |
| S115, S116 | Table 38 | g36_deploy_400m_s2 hard-design falls | 0.2222 | 0.22 | 0.006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s0 hard-design falls | 0.2847 | 0.285 | 0.0006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s1 hard-design falls | 0.2917 | 0.292 | 0.0006 |
| S114 addendum | Table 38 | g35_mref_rsad_hardw_s2 hard-design falls | 0.2778 | 0.278 | 0.0006 |
| S114 | Table 39 | g26 tof18 success | 0.8447 | 0.845 | 0.0006 |
| S114 | Table 39 | g26 tof18 heading_rate_rms | 1.7386 | 1.74 | 0.006 |
| S114 | Table 39 | g26 oracle success | 0.8864 | 0.886 | 0.0006 |
| S114 | Table 39 | g32 tof18 success | 0.8523 | 0.852 | 0.0006 |
| S114 | Table 39 | g32 tof18 fall | 0.0777 | 0.078 | 0.0006 |
| S114 | Table 39 | g32 tof18 heading_rate_rms | 1.2927 | 1.29 | 0.006 |
| S114 | Table 39 | g32 tof18_pp success | 0.5436 | 0.544 | 0.0006 |
| S114 | Table 39 | g32 oracle success | 0.9261 | 0.926 | 0.0006 |
| S114 | Table 39 | g34 tof18 success | 0.8769 | 0.877 | 0.0006 |
| S114 | Table 39 | g34 tof18 fall | 0.0795 | 0.08 | 0.0006 |
| S114 | Table 39 | g34 tof18 stuck | 0.0436 | 0.044 | 0.0006 |
| S114 | Table 39 | g34 tof18 heading_rate_rms | 1.3058 | 1.31 | 0.006 |
| S114 | Table 39 | g34 tof18 cmd_change_mean | 0.3996 | 0.4 | 0.006 |
| S114 | Table 39 | g34 tof18_pp success | 0.5568 | 0.557 | 0.0006 |
| S114 | Table 39 | g34 tof18_pp fall | 0.4129 | 0.413 | 0.0006 |
| S114 | Table 39 | g34 oracle success | 0.9375 | 0.938 | 0.0006 |
| S114 | Table 39 | g34 oracle fall | 0.0606 | 0.061 | 0.0006 |
| S114 | Table 39 | g34 oracle stuck | 0.0019 | 0.002 | 0.0006 |
| S114 | Table 39 | g34raw tof18 success | 0.8807 | 0.881 | 0.0006 |
| S114 | Table 39 | g34raw tof18 fall | 0.0701 | 0.07 | 0.0006 |
| S114 | Table 39 | g34raw tof18 heading_rate_rms | 1.7865 | 1.79 | 0.006 |
| S114 | Table 39 | g34raw oracle success | 0.9148 | 0.915 | 0.0006 |
| S114 | Table 39 | g34raw oracle fall | 0.0833 | 0.083 | 0.0006 |
| S114 addendum | Table 39 | median time to goal, g34 with layer | 9.2200 | 9.2 | 0.06 |
| S114 addendum | Table 39 | median time to goal, g34 without layer | 8.9300 | 8.9 | 0.06 |
| S114 | text | wobble reduction about a quarter | 0.2489 | 0.25 | 0.03 |
| S114 addendum | text | faster layer, pushes: reached | 0.5133 | 0.513 | 0.0006 |
| S114 addendum | text | faster layer, pushes: fell | 0.4545 | 0.455 | 0.0006 |
| S109 | Table 40 | g26 blind_40s success | 0.0777 | 0.078 | 0.0006 |
| S109 | Table 40 | g26 blind_40s fall | 0.9223 | 0.922 | 0.0006 |
| S109 | Table 40 | g26 bump_v15_40s success | 0.5473 | 0.547 | 0.0006 |
| S109 | Table 40 | g26 bump_v15_40s fall | 0.2367 | 0.237 | 0.0006 |
| S109 | Table 40 | g26 bump_v15_40s stuck | 0.2159 | 0.216 | 0.0006 |
| S115 addendum | Table 40 | g34 blind_40s success | 0.0170 | 0.017 | 0.0006 |
| S115 addendum | Table 40 | g34 blind_40s fall | 0.9830 | 0.983 | 0.0006 |
| S115 addendum | Table 40 | g34 bump_v15_40s success | 0.6345 | 0.634 | 0.0006 |
| S115 addendum | Table 40 | g34 bump_v15_40s fall | 0.2614 | 0.261 | 0.0006 |
| S115 addendum | Table 40 | g34 bump_v15_40s stuck | 0.1042 | 0.104 | 0.0006 |
| S114 addendum | Table 41 | full controller falls | 0.0682 | 0.068 | 0.0006 |
| S114 addendum | Table 41 | LQR alone falls | 0.3239 | 0.324 | 0.0006 |
| S114 addendum | Table 41 | kb025 full | 0.3750 | 0.38 | 0.006 |
| S114 addendum | Table 41 | kb025 LQR | 0.9583 | 0.96 | 0.006 |
| S114 addendum | Table 41 | bal4 full | 0.2292 | 0.23 | 0.006 |
| S114 addendum | Table 41 | bal4 LQR | 0.3750 | 0.38 | 0.006 |
| S114 addendum | Table 41 | tau6 full | 0.1250 | 0.12 | 0.006 |
| S114 addendum | Table 41 | tau6 LQR | 0.1458 | 0.15 | 0.006 |
| S114 addendum | Table 41 | others full, max | 0.0208 | 0.02 | 0.006 |
| S114 addendum | Table 41 | others LQR, min | 0.1667 | 0.17 | 0.006 |
| S114 addendum | Table 41 | others LQR, max | 0.3542 | 0.35 | 0.006 |
| S116 | Table 41 | no disturbances | 0.0741 | 0.074 | 0.0006 |
| S116 | Table 41 | short pushes only | 0.0784 | 0.078 | 0.0006 |
| S116 addendum | Table 41 | pulls 0.75 to 3 N | 0.0784 | 0.078 | 0.0006 |
| S116 addendum | Table 41 | pulls 1.5 to 6 N | 0.1395 | 0.139 | 0.0006 |
| S116 addendum | Table 41 | pulls 2.25 to 9 N | 0.2207 | 0.221 | 0.0006 |
| S116 | Table 41 | pulls 3 to 12 N | 0.3608 | 0.361 | 0.0006 |
| S116 addendum | Table 41 | full pulls, bal4 | 0.1813 | 0.18 | 0.006 |
| S116 addendum | Table 41 | full pulls, tau6 | 0.4656 | 0.47 | 0.006 |
| S116 addendum | Table 41 | full pulls, zhigh | 0.1688 | 0.17 | 0.006 |
| S116 addendum | Table 41 | full pulls, kb025 | 0.5938 | 0.59 | 0.006 |
| S116 addendum | Table 41 | full pulls, bal025 | 0.9344 | 0.93 | 0.006 |
| S92 addendum | Table 1 LQR alone | falls @0.3 | 0.0000 | 0.0 | 0.006 |
| S92 addendum | Table 1 LQR alone | err @0.3 | 0.0140 | 0.014 | 0.0006 |
| S92 addendum | Table 1 LQR alone | achieved @0.3 | 1.0000 | 1.0 | 0.006 |
| S92 addendum | Table 1 LQR alone | falls @0.6 | 0.0000 | 0.0 | 0.006 |
| S92 addendum | Table 1 LQR alone | err @0.6 | 0.0170 | 0.017 | 0.0006 |
| S92 addendum | Table 1 LQR alone | achieved @0.6 | 1.0000 | 1.0 | 0.006 |
| S92 addendum | Table 1 LQR alone | falls @0.9 | 0.9100 | 0.89 | 0.006 |
| S92 | methods note | 0.3 m/s recorded falls (border tiles) | 0.3270 | 0.33 | 0.006 |
| S92 | methods note | share at the map edge | 0.7411 | 0.75 | 0.011 |
| S99 addendum | Table 30 (seed 0, friction 0.01) | s0.75 all tiles | 0.4617 | 0.46 | 0.006 |
| S99 addendum | Table 30 (seed 0, friction 0.01) | s0.75 step up larger | 1.0000 | 1.0 | 0.006 |
| S99 addendum | Table 30 (seed 0, friction 0.01) | s0.75 bump larger | 1.0000 | 1.0 | 0.006 |
| S99 addendum | Table 30 (seed 1, friction 0.01) | s0.75 all tiles | 0.4648 | 0.46 | 0.006 |
| S99 addendum | Table 30 (seed 1, friction 0.01) | s0.75 step up larger | 1.0000 | 1.0 | 0.006 |
| S99 addendum | Table 30 (seed 1, friction 0.01) | s0.75 bump larger | 0.9844 | 0.98 | 0.006 |
| S99 addendum | Table 30 (seed 0, friction 0.01) | nom all tiles | 0.3125 | 0.31 | 0.006 |
| S99 addendum | Table 30 (seed 0, friction 0.01) | nom step up larger | 0.9273 | 0.93 | 0.006 |
| S99 addendum | Table 30 (seed 0, friction 0.01) | nom bump larger | 0.9531 | 0.95 | 0.006 |
| S99 addendum | Table 30 (seed 1, friction 0.01) | nom all tiles | 0.3367 | 0.34 | 0.006 |
| S99 addendum | Table 30 (seed 1, friction 0.01) | nom step up larger | 0.9455 | 0.95 | 0.006 |
| S99 addendum | Table 30 (seed 1, friction 0.01) | nom bump larger | 0.9531 | 0.95 | 0.006 |
| S99 addendum | Table 30 (seed 0, friction 0.01) | s1.5 all tiles | 0.2016 | 0.2 | 0.006 |
| S99 addendum | Table 30 (seed 0, friction 0.01) | s1.5 step up larger | 0.5636 | 0.56 | 0.006 |
| S99 addendum | Table 30 (seed 0, friction 0.01) | s1.5 bump larger | 0.4844 | 0.48 | 0.006 |
| S99 addendum | Table 30 (seed 1, friction 0.01) | s1.5 all tiles | 0.2109 | 0.21 | 0.006 |
| S99 addendum | Table 30 (seed 1, friction 0.01) | s1.5 step up larger | 0.6545 | 0.65 | 0.006 |
| S99 addendum | Table 30 (seed 1, friction 0.01) | s1.5 bump larger | 0.5156 | 0.52 | 0.006 |
| S99 addendum | Table 30 (seed 0, friction 0.01) | s2.0 all tiles | 0.1289 | 0.13 | 0.006 |
| S99 addendum | Table 30 (seed 0, friction 0.01) | s2.0 step up larger | 0.5636 | 0.56 | 0.006 |
| S99 addendum | Table 30 (seed 0, friction 0.01) | s2.0 bump larger | 0.4219 | 0.42 | 0.006 |
| S99 addendum | Table 30 (seed 1, friction 0.01) | s2.0 all tiles | 0.1266 | 0.13 | 0.006 |
| S99 addendum | Table 30 (seed 1, friction 0.01) | s2.0 step up larger | 0.6000 | 0.6 | 0.006 |
| S99 addendum | Table 30 (seed 1, friction 0.01) | s2.0 bump larger | 0.4062 | 0.41 | 0.006 |
| S95 | Table 30 (0.5 scale, friction 0.001) | s0.5 step_up smaller / larger | 0.9041 | 0.9 | 0.006 |
| S95 | Table 30 (0.5 scale, friction 0.001) | s0.5 step_up larger | 1.0000 | 1.0 | 0.006 |
| S95 | Table 30 (0.5 scale, friction 0.001) | s0.5 bump smaller / larger | 0.9844 | 0.98 | 0.006 |
| S95 | Table 30 (0.5 scale, friction 0.001) | s0.5 bump larger | 1.0000 | 1.0 | 0.006 |
| S95 | Table 30 (0.5 scale, friction 0.001) | s0.5 stairs smaller / larger | 0.9375 | 0.94 | 0.006 |
| S95 | Table 30 (0.5 scale, friction 0.001) | s0.5 stairs larger | 1.0000 | 1.0 | 0.006 |
| S95 | Table 30 (0.5 scale, friction 0.001) | s0.5 large_drop smaller / larger | 0.9844 | 0.98 | 0.006 |
| S95 | Table 30 (0.5 scale, friction 0.001) | s0.5 large_drop larger | 0.9844 | 0.98 | 0.006 |
| S95 | Table 30 (0.5 scale, friction 0.001) | s0.5 all tiles | 0.9500 | 0.95 | 0.006 |
| S99 addendum | text | large robots lose overall, min | 0.0539 | 0.05 | 0.011 |
| S99 addendum | text | large robots lose overall, max | 0.0566 | 0.06 | 0.011 |
| S99 addendum | text | large robots lose at large steps, up to | 0.1455 | 0.15 | 0.011 |
| S95 addendum | text | 50 percent point, smallest (fraction of r) | 0.1720 | 0.18 | 0.03 |
| S95 addendum | text | 50 percent point, largest (fraction of r) | 0.2884 | 0.22 | 0.03 |
| S95 addendum | text | 50 percent point, mean (fraction of r) | 0.2364 | 0.2 | 0.02 |
| S95 addendum | text | limit at mu 0.4 (fraction of r) | 0.0715 | 0.07 | 0.006 |
| S99 close-out files | Table 32 | scale seed 0 goal undisturbed (battery_mu0.01.json) | 0.7371 | 0.74 | 0.006 |
| S99 close-out files | Table 32 | scale seed 0 goal disturbed | 0.4737 | 0.47 | 0.006 |
| S99 close-out files | Table 32 | scale seed 1 goal undisturbed (battery_mu0.01.json) | 0.6964 | 0.7 | 0.006 |
| S99 close-out files | Table 32 | scale seed 1 goal disturbed | 0.4366 | 0.44 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s1 standing all | 0.0267 | 0.027 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s1 standing kb4 | 0.0000 | 0.0 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s1 standing bal025 | 0.0000 | 0.0 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s1 standing kb025 | 0.2406 | 0.24 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s1 standing bal4 | 0.0531 | 0.05 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s1 driving all | 0.0631 | 0.063 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s1 driving kb025 | 0.5031 | 0.5 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s1 driving bal4 | 0.1844 | 0.18 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s1 driving nom | 0.0031 | 0.0 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s2 standing all | 0.0392 | 0.039 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s2 standing kb4 | 0.0000 | 0.0 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s2 standing bal025 | 0.0000 | 0.0 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s2 standing kb025 | 0.2875 | 0.29 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s2 standing bal4 | 0.1281 | 0.13 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s2 driving all | 0.0747 | 0.075 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s2 driving kb025 | 0.5406 | 0.54 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s2 driving bal4 | 0.2250 | 0.23 | 0.006 |
| S98 addendum | Table 34 | g18_ret_rs025_s2 driving nom | 0.0156 | 0.02 | 0.006 |
| S99 | Table 34 | g26_ret_rs025_mu01_s0 standing all | 0.0281 | 0.028 | 0.006 |
| S99 | Table 34 | g26_ret_rs025_mu01_s0 standing kb025 | 0.2719 | 0.27 | 0.006 |
| S99 | Table 34 | g26_ret_rs025_mu01_s0 standing bal4 | 0.0375 | 0.04 | 0.006 |
| S99 | Table 34 | g26_ret_rs025_mu01_s0 driving all | 0.0622 | 0.062 | 0.006 |
| S99 | Table 34 | g26_ret_rs025_mu01_s0 driving kb025 | 0.4844 | 0.48 | 0.006 |
| S99 | Table 34 | g26_ret_rs025_mu01_s0 driving bal4 | 0.2000 | 0.2 | 0.006 |
| S99 | Table 34 | g26_ret_rs025_mu01_s0 non-finite | 0.0000 | 0.0 | 0.0006 |
| S99 | Table 34 | g26_ret_rs025_mu01_s1 standing all | 0.0301 | 0.03 | 0.006 |
| S99 | Table 34 | g26_ret_rs025_mu01_s1 standing kb025 | 0.2750 | 0.28 | 0.006 |
| S99 | Table 34 | g26_ret_rs025_mu01_s1 standing bal4 | 0.0563 | 0.06 | 0.006 |
| S99 | Table 34 | g26_ret_rs025_mu01_s1 non-finite | 0.0000 | 0.0 | 0.0006 |
| S99 close-out | text | g26 goal DR 1 u min | 0.7335 | 0.73 | 0.006 |
| S99 close-out | text | g26 goal DR 1 u max | 0.7443 | 0.74 | 0.006 |
| S99 close-out | text | g26 goal DR 1 d min | 0.4460 | 0.45 | 0.006 |
| S99 close-out | text | g26 goal DR 1 d max | 0.4597 | 0.46 | 0.006 |
| S99 close-out | text | g26 vel DR 1 u min | 0.3352 | 0.34 | 0.006 |
| S99 close-out | text | g26 vel DR 1 u max | 0.3392 | 0.34 | 0.006 |
| S99 close-out | text | g26 vel DR 1 d min | 0.5097 | 0.51 | 0.006 |
| S99 close-out | text | g26 vel DR 1 d max | 0.5301 | 0.53 | 0.006 |
| S99 close-out | text | g26 goal DR 1.75 u min | 0.7176 | 0.72 | 0.006 |
| S99 close-out | text | g26 goal DR 1.75 u max | 0.7244 | 0.72 | 0.006 |
| S99 close-out | text | g26 goal DR 1.75 d min | 0.4097 | 0.41 | 0.006 |
| S99 close-out | text | g26 goal DR 1.75 d max | 0.4199 | 0.42 | 0.006 |
| S99 close-out | text | g26 vel DR 1.75 u min | 0.3415 | 0.34 | 0.006 |
| S99 close-out | text | g26 vel DR 1.75 u max | 0.3602 | 0.36 | 0.006 |
| S99 close-out | text | g26 vel DR 1.75 d min | 0.5483 | 0.55 | 0.006 |
| S99 close-out | text | g26 vel DR 1.75 d max | 0.5636 | 0.56 | 0.006 |
| S99 close-out | text | g26 goal DR 2.5 u min | 0.6795 | 0.68 | 0.006 |
| S99 close-out | text | g26 goal DR 2.5 u max | 0.6909 | 0.69 | 0.006 |
| S99 close-out | text | g26 goal DR 2.5 d min | 0.3324 | 0.33 | 0.006 |
| S99 close-out | text | g26 goal DR 2.5 d max | 0.3585 | 0.36 | 0.006 |
| S99 close-out | text | g26 vel DR 2.5 u min | 0.3750 | 0.38 | 0.006 |
| S99 close-out | text | g26 vel DR 2.5 u max | 0.3937 | 0.39 | 0.006 |
| S99 close-out | text | g26 vel DR 2.5 d min | 0.6011 | 0.6 | 0.006 |
| S99 close-out | text | g26 vel DR 2.5 d max | 0.6227 | 0.62 | 0.006 |
| S99 close-out | text | g26 flat achieved 0.3 min | 0.8464 | 0.85 | 0.006 |
| S99 close-out | text | g26 flat achieved 0.3 max | 0.9265 | 0.93 | 0.006 |
| S99 close-out | text | g26 flat achieved 0.6 min | 0.9509 | 0.95 | 0.006 |
| S99 close-out | text | g26 flat achieved 0.6 max | 0.9754 | 0.98 | 0.006 |
| S99 close-out | text | g26 flat achieved 0.9 min | 0.9894 | 0.99 | 0.006 |
| S99 close-out | text | g26 flat achieved 0.9 max | 0.9993 | 1.0 | 0.006 |
| S99 close-out | text | g26 flat falls max | 0.0000 | 0.0 | 0.006 |
| S99 close-out | Table 34 | g26 standing min | 0.0250 | 0.025 | 0.0006 |
| S99 close-out | Table 34 | g26 standing max | 0.0301 | 0.03 | 0.0006 |
| S99 close-out | Table 34 | g26 driving min | 0.0622 | 0.062 | 0.0006 |
| S99 close-out | Table 34 | g26 driving max | 0.0685 | 0.068 | 0.0006 |
| S99 close-out | Table 34 | g26_ret_rs025_mu01_s2 standing all | 0.0250 | 0.025 | 0.006 |
| S99 close-out | Table 34 | g26_ret_rs025_mu01_s2 standing kb025 | 0.2375 | 0.24 | 0.006 |
| S99 close-out | Table 34 | g26_ret_rs025_mu01_s2 standing bal4 | 0.0375 | 0.04 | 0.006 |
| S99 close-out | Table 34 | g26_ret_rs025_mu01_s2 driving all | 0.0639 | 0.064 | 0.006 |
| S99 close-out | Table 34 | g26_ret_rs025_mu01_s2 driving kb025 | 0.4906 | 0.49 | 0.006 |
| S99 close-out | Table 34 | g26_ret_rs025_mu01_s2 driving bal4 | 0.2062 | 0.21 | 0.006 |
| S99 close-out | Table 34 | g26_ret_rs025_mu01_s2 non-finite | 0.0000 | 0.0 | 0.0006 |
| S6/S7/S10/S11 (paper tab:tracking) | Table 0 | lqr ratio 0.5 | 1.0369 | 1.04 | 0.006 |
| S6/S7/S10/S11 (paper tab:tracking) | Table 0 | lqr ratio 1.0 | 1.0640 | 1.06 | 0.006 |
| S6/S7/S10/S11 (paper tab:tracking) | Table 0 | trk_f_s0 ratio 0.5 | 0.9809 | 0.98 | 0.006 |
| S6/S7/S10/S11 (paper tab:tracking) | Table 0 | trk_f_s0 ratio 1.0 | 0.9142 | 0.91 | 0.006 |
| S6/S7/S10/S11 (paper tab:tracking) | Table 0 | trk_f_s1 ratio 0.5 | 1.0851 | 1.09 | 0.006 |
| S6/S7/S10/S11 (paper tab:tracking) | Table 0 | trk_f_s1 ratio 1.0 | 0.9499 | 0.95 | 0.006 |
| S6/S7/S10/S11 (paper tab:tracking) | Table 0 | trk_g_s0 ratio 0.5 | 0.9928 | 0.99 | 0.006 |
| S6/S7/S10/S11 (paper tab:tracking) | Table 0 | trk_g_s0 ratio 1.0 | 0.8667 | 0.87 | 0.006 |
| S6/S7/S10/S11 (paper tab:tracking) | Table 0 | robs_b_s0 ratio 0.5 | 0.4993 | 0.5 | 0.006 |
| S6/S7/S10/S11 (paper tab:tracking) | Table 0 | robs_b_s0 ratio 1.0 | 0.5091 | 0.51 | 0.006 |
| S6/S7/S10/S11 (paper tab:tracking) | Table 0 | res_c_s0 ratio 0.5 | 1.0292 | 1.03 | 0.006 |
| S6/S7/S10/S11 (paper tab:tracking) | Table 0 | res_c_s0 ratio 1.0 | 1.0534 | 1.05 | 0.006 |
| S7 | text | trk_h falls/30 | 3.0000 | 3 | 0.5 |
| S10 | text | res_c falls/30 | 3.0000 | 3 | 0.5 |
| S10 | text | res_c heading err | 0.0191 | 0.019 | 0.001 |
| S10 | text | res_d random falls | 15.0000 | 15 | 0.5 |
| S20 | text | Warp 4096 worlds / one CPU process | 260.4821 | 260 | 1.5 |
| S46 | Table 6 | mean_baseline R2 | -0.0003 | 0.0 | 0.006 |
| S46 | Table 6 | mean_baseline MAE | 3.8746 | 3.87 | 0.006 |
| S46 | Table 6 | mean_baseline rise F1 | 0.0000 | 0.0 | 0.006 |
| S46 | Table 6 | mean_baseline drop F1 | 0.0000 | 0.0 | 0.006 |
| S46 | Table 6 | proprio_only R2 | 0.0151 | 0.02 | 0.006 |
| S46 | Table 6 | proprio_only MAE | 3.9739 | 3.97 | 0.006 |
| S46 | Table 6 | proprio_only rise F1 | 0.2519 | 0.25 | 0.006 |
| S46 | Table 6 | proprio_only drop F1 | 0.2045 | 0.2 | 0.006 |
| S46 | Table 6 | depth+proprio R2 | 0.5526 | 0.55 | 0.006 |
| S46 | Table 6 | depth+proprio MAE | 2.1910 | 2.19 | 0.006 |
| S46 | Table 6 | depth+proprio rise F1 | 0.6178 | 0.62 | 0.006 |
| S46 | Table 6 | depth+proprio drop F1 | 0.6672 | 0.67 | 0.006 |
| S46 | Table 6 | depth_only_policy_shaped R2 | 0.5961 | 0.6 | 0.006 |
| S46 | Table 6 | depth_only_policy_shaped MAE | 1.9279 | 1.93 | 0.006 |
| S46 | Table 6 | depth_only_policy_shaped rise F1 | 0.6969 | 0.7 | 0.006 |
| S46 | Table 6 | depth_only_policy_shaped drop F1 | 0.7335 | 0.73 | 0.006 |
| S74 | text | soft gain undisturbed (about +0.03 to +0.05) | 0.0466 | 0.04 | 0.02 |
| S74 | text | seed spread of goal success about 0.02 | 0.0222 | 0.02 | 0.011 |
