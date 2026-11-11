"""Idempotently add the shared <head> pieces to every top-level page of site/: the theme/favicon/fonts block (head:common) where missing,
and canonical + Open Graph + Twitter card tags (og:image is assets/img/social-card.jpg, 1200x630). Also regenerates sitemap.xml.

    python scripts/site/add_meta.py
"""
import html
import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[2] / "site"
BASE = "https://n3b3x.github.io/openballbot-rl/"
COMMON = """<!-- head:common -->
<script>try{var t=localStorage.getItem('theme');document.documentElement.dataset.theme=t||(matchMedia('(prefers-color-scheme: light)').matches?'light':'dark')}catch(e){}</script>
<meta name="color-scheme" content="dark light">
<link rel="icon" href="assets/img/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<link rel="stylesheet" href="assets/css/tokens.css">
<!-- /head:common -->
"""

NOJS = "<!-- nojs --><noscript><nav class=\"nojs\" aria-label=\"Main (JavaScript is off)\">" + "".join(f'<a href="{h}">{t}</a>' for h, t in [("index.html", "Overview"), ("learn/index.html", "Learn"), ("generalist.html", "Results"), ("videos.html", "Videos"), ("design.html", "Design guidelines"), ("deploy.html", "ROS 2"), ("math.html", "Physics"), ("architecture.html", "Controller"), ("reproduce.html", "Reproduce"), ("glossary.html", "Glossary")]) + "</nav></noscript><!-- /nojs -->\n"

pages = sorted(p for p in SITE.glob("*.html") if p.name != "404.html")
for p in pages:
    s = p.read_text(encoding="utf8")
    if "<!-- head:common -->" in s:
        s = re.sub(r"<!-- head:common -->.*?<!-- /head:common -->\n", lambda m: COMMON, s, count=1, flags=re.S)
    else:
        s = re.sub(r'(<meta name="viewport"[^>]*>\n)', lambda m: m.group(1) + COMMON, s, count=1)
    title = html.unescape(re.search(r"<title>(.*?)</title>", s, re.S).group(1).strip())
    desc = re.search(r'<meta name="description" content="(.*?)">', s, re.S).group(1)
    url = BASE if p.name == "index.html" else BASE + p.name
    og = f"""<!-- og -->
<link rel="canonical" href="{url}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Ballbot control research">
<meta property="og:title" content="{html.escape(title, quote=True)}">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{BASE}assets/img/social-card.jpg">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="A simulated ballbot: a cylindrical robot balanced on a basketball-textured ball, with the title of the study.">
<meta name="twitter:card" content="summary_large_image">
<!-- /og -->
"""
    s = re.sub(r"<!-- og -->.*?<!-- /og -->\n", "", s, flags=re.S)
    s = re.sub(r'(<meta name="description" content=".*?">\n)', lambda m: m.group(1) + og, s, count=1, flags=re.S)
    s = re.sub(r"<!-- nojs -->.*?<!-- /nojs -->\n", "", s, flags=re.S)
    s = re.sub(r"(<body[^>]*>\n)", lambda m: m.group(1) + NOJS, s, count=1)
    p.write_text(s, encoding="utf8")

urls = [BASE if p.name == "index.html" else BASE + p.name for p in pages] + [BASE + q.relative_to(SITE).as_posix() for q in sorted((SITE / "learn").glob("*.html"))]
(SITE / "sitemap.xml").write_text('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + "".join(f"  <url><loc>{u}</loc></url>\n" for u in urls) + "</urlset>\n", encoding="utf8")
(SITE / "robots.txt").write_text(f"User-agent: *\nAllow: /\n\nSitemap: {BASE}sitemap.xml\n", encoding="utf8")
print(len(pages), "pages;", len(urls), "sitemap urls")
