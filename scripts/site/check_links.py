"""Link checker for site/: every internal href/src/poster in every HTML file must resolve to a file, and every #fragment to an id in the target page.

    python scripts/site/check_links.py            # exit code 1 if anything is broken
    python scripts/site/check_links.py --external # also list the external hosts used (no network access)

Also flags em dashes in visible text and pages that lack a <title>, viewport meta or h1. Links injected by assets/js/layout.js (navigation, footer)
are checked from the ROUTES list below, because the checker reads static HTML only.
"""
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

SITE = Path(__file__).resolve().parents[2] / "site"
# pages linked from the shared chrome (layout.js)
ROUTES = ["index.html", "learn/index.html", "generalist.html", "videos.html", "design.html", "deploy.html", "math.html", "architecture.html", "bounds.html", "lessons.html",
          "design-space.html", "live.html", "reproduce.html", "story.html", "method.html", "results.html", "engines.html", "literature.html", "glossary.html", "assets/img/icons.svg", "assets/img/favicon.svg"]


class P(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.ids, self.text, self.title, self.h1, self.viewport, self._skip, self._in_title = [], set(), [], "", 0, False, 0, False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "a" and a.get("name"):
            self.ids.add(a["name"])
        for k in ("href", "src", "poster", "data-src", "data-poster"):
            if a.get(k) and not (k.startswith("data-") and not re.search(r"\.(mp4|jpg|png|svg|json)$", a[k])):
                self.links.append((tag, k, a[k]))
        if tag == "meta" and a.get("name") == "viewport":
            self.viewport = True
        if tag == "h1":
            self.h1 += 1
        if tag in ("script", "style"):
            self._skip += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, d):
        if self._in_title:
            self.title += d
        elif not self._skip:
            self.text.append(d)


def parse(path):
    p = P()
    p.feed(path.read_text(encoding="utf8", errors="replace"))
    return p


def main():
    pages = sorted(SITE.rglob("*.html"))
    info = {f: parse(f) for f in pages}
    bad, ext, warn = [], set(), []
    for f, p in info.items():
        rel = f.relative_to(SITE)
        for tag, k, v in p.links:
            if v.startswith(("mailto:", "tel:", "javascript:", "data:")) or "{{" in v:
                continue
            u = urlparse(v)
            if u.scheme in ("http", "https"):
                ext.add(u.netloc)
                continue
            if u.scheme or v.startswith("//"):
                continue
            target = f if not u.path else (f.parent / unquote(u.path)).resolve()
            if u.path.endswith("/"):
                target = target / "index.html"
            if not target.exists():
                bad.append(f"{rel}: <{tag} {k}> {v} -> missing file")
                continue
            if u.fragment and target.suffix == ".html":
                tp = info.get(target) or parse(target)
                if unquote(u.fragment) not in tp.ids and unquote(u.fragment) != "top":
                    bad.append(f"{rel}: {v} -> no id '{u.fragment}' in {target.relative_to(SITE)}")
        text = " ".join(p.text)
        if "—" in text or "–" in text:
            warn.append(f"{rel}: em or en dash in visible text")
        if rel.parts[0] != "learn" or True:
            if not p.title.strip():
                warn.append(f"{rel}: no <title>")
            if not p.viewport:
                warn.append(f"{rel}: no viewport meta")
            if p.h1 != 1 and f.name != "index.html":
                warn.append(f"{rel}: {p.h1} h1 elements")
    for r in ROUTES:
        if not (SITE / r).exists():
            bad.append(f"layout.js route {r} -> missing file")
    print(f"{len(pages)} pages checked")
    for b in bad:
        print("BROKEN", b)
    for w in warn:
        print("WARN  ", w)
    if "--external" in sys.argv:
        print("external hosts:", ", ".join(sorted(ext)))
    print(f"{len(bad)} broken links, {len(warn)} warnings")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
