# Site design system and ownership

Static site (no build step) for GitHub Pages. Vanilla ES modules, CSS tokens in `assets/css/tokens.css`, shared chrome in
`assets/js/layout.js`. Every page: `<link rel="stylesheet" href="assets/css/site.css">`, `<script type="module" src="assets/js/layout.js">`.
Page scaffold: `<body><main>...</main></body>`; layout.js adds the skip link, nav, prev/next pager and footer. Use `.wrap`, `section.block`,
`.eyebrow`, `.lede`, `.card`, `.grid.g2/.g3`, `.chip`, `.callout`, `.reveal` and the components below (see site.css).

Colour semantics (do not reuse for other meanings): `--model` blue = LQR / derived math; `--learned` violet = PPO / neural;
`--hybrid` teal = the interleaved controller; `--warn` amber and `--bad` red = limits and failures; `--good` green = success.
Typography: Fraunces (display), Inter (body), JetBrains Mono (numbers, code). Charts: draw with SVG or Canvas using the CSS
variables so they follow the light/dark theme (listen for the `themechange` event on document). Respect prefers-reduced-motion.
Third-party libs only from cdnjs.cloudflare.com / cdn.jsdelivr.net / unpkg.com with PINNED versions. No tracking.
Layout: mobile first, 16px gutters, no horizontal page scroll at 375 px (wrap wide tables in `.table > .scroll` or `.tablewrap`).
Accessible: alt text, captions, keyboard focus, contrast.
Truthfulness: every number on the site must come from `site/data/*.json` (generated from repo CSV/logs) or be cited to a
file in the repo (log entries by S-number, in a `data-src` attribute or a footnote); label simulation-only results as such;
no hardware claims; no invented citations. No em dashes in copy.

## Reading order (2026-11-08 brief)

The nav expresses it with numbers: 1 Start here (index), 2 Physics (math), 3 Controller (architecture), 4 Results (generalist),
5 Rules and bounds (bounds), 6 Lessons (lessons), 7 Videos. Everything else sits under "More": Design space explorer, Live policy,
Literature, Reproduce, then The journey (story), Setup and method (method), Data tables (results), Engines, Glossary.
Pages in the order also get a prev/next pager above the footer. To change either list edit `ORDER` / `MORE` in `assets/js/layout.js`.

Every page should open with: eyebrow, h1, one-paragraph `.lede`, then a `.tldr` box (with a `.readtime` chip), then, on long pages,
a `nav.pagetoc`. Short lead-in first, detail inside `details.more`. Visuals before tables. Define a term on first use with `.term`.

## Shared components (site.css + layout.js)

Each line: what it is, then a one-line usage example. All are theme-aware and need no page CSS.

- **TL;DR box** `.tldr`: bordered summary at the top of a page. Prefix label is "TL;DR" (override with `data-label`). Put a `.readtime` inside for the chip.
  `<aside class="tldr"><span class="readtime"></span><ul><li><b>Claim.</b> One sentence.</li></ul><span class="fine">Cited to S84.</span></aside>`
- **Reading-time chip** `.readtime`: layout.js fills every empty `.readtime` with "N min read" (220 words per minute over `<main>`).
  `<span class="readtime"></span>`
- **Defined term** `.term[data-def]`: dotted term that shows a short definition on hover, keyboard focus and first tap, and links to the glossary.
  Styling is keyed on `data-def`, so arch.css's unrelated `.term` grid is untouched. Keep definitions to one sentence.
  `<a class="term" href="glossary.html#authority" data-def="Scale on the learned correction; 0.25 means at most a quarter of the range.">authority</a>`
  (`<dfn class="term" data-def="...">` works for a term defined in place with no link.)
- **Numbered figure** `.figure`: `<figure class="figure">` numbers its caption "Figure N." automatically (CSS counter over the whole page; `data-num="3"` forces a number).
  Put wide SVG/canvas in `<div class="media">` to get horizontal scroll inside the figure instead of page scroll. `.figure.plain` drops the panel background.
  `<figure class="figure"><img src="figures/x.svg" alt="..."><figcaption><b>Lead sentence.</b> What to see. <span class="src">data/study.json, S84</span></figcaption></figure>`
- **Numbered table** `.table`: caption above, "Table N." prefix, table scrolls inside `.scroll` on phones. Use `td.n` for right-aligned mono numbers.
  `<div class="table"><p class="caption"><b>Goal success</b>, 3 seeds.</p><div class="scroll"><table>...</table></div><span class="src">S84</span></div>`
- **Key-number card** `.keynum` (in a `.keynums` grid): big number, one-line meaning, optional detail, and the log entry from `data-src` rendered as "log S84".
  Colour variants `.model .learned .hybrid .good .warn .bad`.
  `<div class="keynums"><div class="keynum hybrid" data-src="S84"><b>0.71 <small>to 0.76</small></b><span class="what">goal success, undisturbed</span><p>3 seeds, 11 designs.</p></div></div>`
- **Progressive disclosure** `details.more`: collapsed detail under a short lead-in (same look as story.css's version, now shared).
  `<details class="more"><summary>The equations</summary><div><p>...</p></div></details>`
- **Sticky in-page TOC** `nav.pagetoc`: place an empty one after the TL;DR; layout.js fills it from every `h2[id]` in `<main>` (ids are added
  when missing; `data-level="h3"` includes h3), sticks it under the nav and highlights the current section. A pre-filled `<ol>` is left as is.
  `<nav class="pagetoc" aria-label="On this page"></nav>`
- **Source footnotes** `.src-note`, `sup.ref`: small mono provenance line, and an inline superscript reference to a footnote id.
  `<p class="src-note">Protocol: MuJoCo-Warp GPU, 100 Hz; see <a href="method.html">Setup</a>.</p>` and `<sup class="ref"><a href="#src-S84" data-src="S84">S84</a></sup>`
- **Site chrome** (automatic): `.skip` link, `.nav` with numbered reading-order links, `details.more-menu` More dropdown, mobile `Menu` button (below 960 px),
  theme toggle (dispatches `themechange`), `.pager` prev/next, `footer.site`.
- Also shared: `.chip.warn`, `.num` (mono numbers), `.sim` (small "simulation" label), `.visually-hidden`, `:focus-visible` ring.

## File ownership (agents must only write inside their own files)

- data/media agent: `scripts/site/*`, `site/data/*`, `site/videos/*`, `site/figures/*`, `videos.html`, `results.html`, `assets/js/charts.js`
- story agent: `story.html`, `literature.html`, `assets/js/story.js`
- architecture agent: `architecture.html`, `assets/js/arch.js`, `assets/img/*.svg` (diagrams, icons)
- math/design agent: `math.html`, `design-space.html`, `assets/js/ballbot-model.js`, `assets/js/math.js`, `assets/js/design.js`
- hero/3D + live agent: `live.html`, `reproduce.html`, `assets/js/hero3d.js`, `assets/js/live.js`, `site/live/*`
- new pages (2026-11-08): `bounds.html`, `lessons.html`, `glossary.html` by their assigned agents
- the lead owns `index.html`, `assets/css/*`, `assets/js/layout.js`, `DESIGN.md`, `.github/workflows/pages.yml`.
Shared CSS needs: put page-specific CSS in `<style>` inside your page or in `assets/css/<page>.css`; do not edit site.css.
Name clashes to avoid in page CSS: `.toc` (reproduce chip row, home.css), `.stat` (site.css and study.css), `.term` (arch.css grid; the shared
component is `.term[data-def]`), `.more` (shared `details.more`).

Generalist-study pages (`method.html`, `generalist.html`): every table, figure and prose number on them is generated by `python site/build_data.py`
(tables inside `<!-- BUILD:name -->` blocks, prose numbers as `{{key:decimals}}` tokens that become `<span class="v" data-k=...>`, figures in `figures/study/`,
paper tables and `\gv{key}` values in `paper_v2/`). Do not hand-edit those blocks or spans; do not put value spans inside KaTeX math. CSS: `assets/css/study.css`.
Evaluation inputs are read from `data/eval_snapshot/` (run `python site/build_data.py --refresh` to re-copy the live files).
