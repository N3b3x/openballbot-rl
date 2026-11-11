# Ballbot course (site/learn)

An interactive course in ten lessons on how a ballbot works and how this project's pipeline was built. Static HTML, CSS and
ES modules, no build step; it is served as part of the GitHub Pages site (`/learn/`). Simulation results only: there is no
physical robot, and every number is read from a repo file or cited to a study-log entry (S-number).

```
learn/
  index.html              landing page: lesson cards, progress (localStorage, try/catch), estimated time
  01-ballbot.html ...     lessons 1 to 10 (goals, explanation, figures, "try it" / quiz with reveal, takeaways, code and log links)
  assets/css/learn.css    all styles; shares only ../assets/css/tokens.css with the rest of the site (dark and light themes)
  assets/js/learn.js      chrome: top bar, theme toggle, progress, pager, quiz buttons, lazy loading of figures
  assets/js/physics.js    planar ballbot model, discrete LQR, RK4 simulation (the model of scripts/study/ballbot_model.py)
  assets/js/ui.js         SVG chart, canvas helpers, sliders, read-outs (no dependencies)
  assets/js/l01.js ... l10.js   one module per lesson, imported only when a figure scrolls near the viewport
  data/learn.json         generated: model, gains, designs, S119 sweep, S108 odometry, parity table, cited log numbers
```

* **Figures** are `<figure class="widget" data-widget="l03:loop">`: module `l03.js`, exported function `loop(el, ctx)`.
* **3D** (lesson 1 only) uses three.js 0.160.0 from jsDelivr via an import map, the same source and version as the landing hero, and
  the shared `../assets/js/robot3d.js` model read-only (a plain pendulum is drawn if that module cannot be loaded).
* **Accessibility:** keyboard-operable controls with labels, `aria-label` on charts and canvases, the planner grid is usable
  with arrow keys and space, no autoplay, `prefers-reduced-motion` respected (Play buttons compute the result at once).
* **Theme:** colours come from the CSS tokens, so charts follow light and dark; canvases redraw on the `themechange` event.

## Regenerating the data and checking

```bash
python scripts/site/build_learn_data.py            # verifies every quoted log number against the log, writes data/learn.json
python scripts/site/build_learn_data.py --check    # only the log citations
python scripts/site/build_learn_data.py --links    # link check (files, anchors, GitHub blob links) and dash check over site/learn/
```

Inputs: `ballbot_gym/models/ballbot_planar_model.json` and `variants/dz_*.json`, `ros2/ballbot_control/config/bundle/g39_goalcentre_s0/bundle.json`,
`outputs/study/specs/{ota_sim,ota_sweep,spec_nominal_sim}.csv` and `spec_nominal.json`, `outputs/study/warp_ppo/odom/odom_results_flat.json`,
`outputs/study/warp_ppo/g34_mref_rsad_db_s0/motion_goal_ref_allflat*.json`, the parity table in `ros2/README.md`, and
`docs/research/tracking_study_log.md` for the quoted study numbers. Values that exist only as prose in the log (authority curve, seed
spread, navigation outcomes) are listed in the script with the exact log text each one is read from; the script fails if the text is gone.

Widgets that are stand-ins say so in their captions: the follower in lesson 5 is a first-order lag, the exposure figure in lesson 9 and
the mixing figure in lesson 4 use toy signals, and the sensor view in lesson 6 is a simple ray cast with the study's sensor settings.
