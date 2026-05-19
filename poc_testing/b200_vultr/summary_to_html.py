#!/usr/bin/env python3
"""Convert summary.md to a styled, single-file summary.html with floating left navbar.

Handles only the markdown subset emitted by aggregate_results.py:
  - h1/h2/h3 headers (and h4 promoted from numbered sections)
  - GitHub-flavored tables (header row + --- separator + body rows)
  - Fenced code blocks (```)
  - Bulleted lists ("- ...")
  - Blockquotes (lines starting with "> ") rendered as callout boxes
  - Inline: **bold**, `code`, [link](url), ![alt](src), <https://autolink>

Dark mode, distinctive section headers, navbar built from h2/h3.

Usage: summary_to_html.py <input.md> <output.html>
"""
import html
import re
import sys


CSS = """
:root {
  --bg:        #0d1117;
  --bg-elev:   #161b22;
  --bg-elev-2: #1c232c;
  --text:      #e6edf3;
  --text-dim:  #b1bac4;
  --text-mute: #7d8590;
  --accent:    #58a6ff;        /* links */
  --accent-2:  #7ee787;         /* highlights / NVIDIA-green-ish */
  --accent-3:  #f0883e;         /* callouts / warnings */
  --accent-4:  #d2a8ff;         /* h2 chrome */
  --border:    #30363d;
  --border-2:  #21262d;
  --code-bg:   #1f2428;
  --code-text: #c9d1d9;
  --quote-bg:  #161e2b;
}

* { box-sizing: border-box; }

html { scroll-behavior: smooth; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 14.5px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}

/* ============ Layout: fixed left navbar + content ============ */
.nav {
  position: fixed;
  top: 0; left: 0; bottom: 0;
  width: 280px;
  padding: 1.25em 1em 2em 1.25em;
  overflow-y: auto;
  background: var(--bg-elev);
  border-right: 1px solid var(--border);
  font-size: 13px;
}
.nav h2 {
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-mute);
  margin: 0 0 0.5em 0;
  border: none;
  padding: 0;
}
.nav ul { list-style: none; padding: 0; margin: 0 0 1em 0; }
.nav li { margin: 0; }
.nav .lvl-h2 > a {
  display: block;
  padding: 0.35em 0.5em;
  margin-top: 0.5em;
  font-weight: 600;
  color: var(--text);
  border-left: 2px solid var(--accent-4);
  text-decoration: none;
}
.nav .lvl-h2 > a:hover { background: var(--bg-elev-2); }
.nav .lvl-h3 > a {
  display: block;
  padding: 0.25em 0.5em 0.25em 1.4em;
  color: var(--text-dim);
  text-decoration: none;
  border-left: 2px solid transparent;
}
.nav .lvl-h3 > a:hover { color: var(--text); background: var(--bg-elev-2); border-left-color: var(--accent); }

.content {
  margin-left: 300px;
  padding: 2em 2em 4em 2em;
  max-width: 1200px;
}

@media (max-width: 900px) {
  .nav { position: static; width: 100%; height: auto; border-right: none; border-bottom: 1px solid var(--border); }
  .content { margin-left: 0; padding: 1.25em; }
}

/* ============ Headings ============ */
h1 {
  font-size: 28px;
  margin: 0 0 0.4em 0;
  padding-bottom: 0.4em;
  border-bottom: 1px solid var(--border);
  color: var(--text);
  letter-spacing: -0.01em;
}

h2 {
  font-size: 22px;
  margin: 2.2em 0 0.6em 0;
  padding: 0.45em 0.8em;
  background: linear-gradient(90deg, var(--bg-elev-2), transparent);
  border-left: 4px solid var(--accent-4);
  color: var(--text);
}

h3 {
  font-size: 17.5px;
  margin: 1.8em 0 0.4em 0;
  padding-bottom: 0.25em;
  border-bottom: 1px solid var(--border-2);
  color: var(--accent-2);
}

h4 {
  font-size: 15.5px;
  margin: 1.4em 0 0.4em 0;
  color: var(--text-dim);
}

/* Anchors next to headings */
h1[id], h2[id], h3[id], h4[id] { scroll-margin-top: 1em; }

p { margin: 0.65em 0; }

a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ============ Tables ============ */
table {
  border-collapse: collapse;
  margin: 1em 0;
  font-size: 13px;
  width: 100%;
  background: var(--bg-elev);
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}
th, td {
  border-bottom: 1px solid var(--border-2);
  padding: 6px 10px;
  text-align: left;
}
th {
  background: var(--bg-elev-2);
  color: var(--text);
  font-weight: 600;
  font-size: 12.5px;
  text-transform: none;
  border-bottom: 1px solid var(--border);
}
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--bg-elev-2); }

/* ============ Lists ============ */
ul, ol { margin: 0.5em 0 1em 0; padding-left: 1.5em; }
li { margin: 0.2em 0; }
li::marker { color: var(--text-mute); }

/* ============ Code ============ */
code {
  background: var(--code-bg);
  color: var(--code-text);
  padding: 1px 5px;
  border-radius: 3px;
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.88em;
}
pre {
  background: var(--code-bg);
  color: var(--code-text);
  padding: 0.9em 1em;
  overflow-x: auto;
  border: 1px solid var(--border-2);
  border-radius: 6px;
  font-size: 12.5px;
  line-height: 1.45;
}
pre code { background: transparent; padding: 0; }

/* ============ Callout (blockquote) ============ */
blockquote {
  margin: 1em 0;
  padding: 0.7em 1em 0.7em 1.1em;
  background: var(--quote-bg);
  border-left: 3px solid var(--accent-3);
  border-radius: 0 5px 5px 0;
  color: var(--text-dim);
}
blockquote p { margin: 0.35em 0; }
blockquote strong { color: var(--text); }

/* ============ Boxed legends and notes ============ */
.legend, .note {
  margin: 1em 0;
  padding: 0.6em 1em 0.7em 1em;
  border-radius: 6px;
  border: 1px solid var(--border-2);
}
.legend {
  background: #131a23;
  border-left: 3px solid var(--accent);    /* blue — informational */
  color: var(--text-dim);
  font-size: 13.5px;
}
.legend strong { color: var(--text); }
.legend ul { margin: 0.25em 0 0 0; padding-left: 1.4em; }
.legend p:first-child, .legend > p:first-child {
  margin-top: 0;
  font-size: 12px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--accent);
}

.note {
  background: var(--quote-bg);
  border-left: 4px solid var(--accent-3);   /* orange — commentary */
  color: var(--text-dim);
}
.note p { margin: 0.4em 0; }
.note strong { color: var(--text); }
.note ul { margin: 0.4em 0; padding-left: 1.4em; }
.note > p:first-child::before {
  content: "📝 ";
  margin-right: 0.15em;
  filter: saturate(1.2);
}

/* ============ Images ============ */
img { max-width: 100%; height: auto; border-radius: 4px; border: 1px solid var(--border-2); background: #fff; }

strong { color: var(--text); }
"""


INLINE_CODE = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*([^*]+)\*\*")
IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
AUTOLINK = re.compile(r"&lt;(https?://[^&\s]+)&gt;")  # already escaped before this regex runs


def slugify(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", "", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s


def inline(text):
    text = html.escape(text)
    text = IMAGE.sub(r'<img src="\2" alt="\1">', text)
    text = LINK.sub(r'<a href="\2">\1</a>', text)
    text = AUTOLINK.sub(r'<a href="\1">\1</a>', text)
    text = INLINE_CODE.sub(r"<code>\1</code>", text)
    text = BOLD.sub(r"<strong>\1</strong>", text)
    return text


def is_separator(row_cells):
    return all(re.fullmatch(r":?-{3,}:?", c.strip()) for c in row_cells)


def split_row(line):
    s = line.strip()
    if s.startswith("|"): s = s[1:]
    if s.endswith("|"):   s = s[:-1]
    return [c.strip() for c in s.split("|")]


def cell_class(cell, align):
    if align == "right":
        return ' class="num"'
    if re.fullmatch(r"[-+]?\d+(\.\d+)?%?", cell):
        return ' class="num"'
    return ""


def aligns_from_sep(cells):
    aligns = []
    for c in cells:
        c = c.strip()
        left = c.startswith(":"); right = c.endswith(":")
        if left and right: aligns.append("center")
        elif right:        aligns.append("right")
        else:              aligns.append("left")
    return aligns


def build_nav(headings):
    """Headings is list of (level, slug, text). Build a hierarchical TOC."""
    parts = ['<nav class="nav">', '<h2>Contents</h2>', '<ul>']
    for level, slug, text in headings:
        cls = f"lvl-h{level}"
        parts.append(f'<li class="{cls}"><a href="#{slug}">{html.escape(text)}</a></li>')
    parts.append('</ul></nav>')
    return "\n".join(parts)


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: summary_to_html.py <input.md> <output.html>\n")
        sys.exit(2)

    lines = open(sys.argv[1]).read().splitlines()

    # First pass — collect headings for nav (h2/h3 only — h4 is too noisy in TOC)
    headings = []
    for line in lines:
        m = re.match(r"^(#{2,3})\s+(.+?)\s*$", line)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            headings.append((level, slugify(text), text))

    body = []
    in_code = False
    in_table = False
    table_buf = []
    in_list = False
    in_quote = False
    in_div = None   # name of the open ::: fence div, or None
    page_title = "B200 / Vultr POC — cluster test summary"

    def close_table():
        nonlocal in_table, table_buf
        if not in_table: return
        header_cells = split_row(table_buf[0])
        aligns = aligns_from_sep(split_row(table_buf[1]))
        body.append('<table>')
        body.append('<thead><tr>')
        for c in header_cells:
            body.append(f"<th>{inline(c)}</th>")
        body.append('</tr></thead><tbody>')
        for row_line in table_buf[2:]:
            cells = split_row(row_line)
            body.append('<tr>')
            for i, c in enumerate(cells):
                a = aligns[i] if i < len(aligns) else "left"
                cls = cell_class(c, a)
                body.append(f"<td{cls}>{inline(c)}</td>")
            body.append('</tr>')
        body.append('</tbody></table>')
        in_table = False
        table_buf = []

    def close_list():
        nonlocal in_list
        if in_list:
            body.append('</ul>')
            in_list = False

    def close_quote():
        nonlocal in_quote
        if in_quote:
            body.append('</blockquote>')
            in_quote = False

    for line in lines:
        # Fenced code block
        if line.startswith("```"):
            close_table(); close_list(); close_quote()
            if in_code:
                body.append('</code></pre>')
                in_code = False
            else:
                body.append('<pre><code>')
                in_code = True
            continue
        if in_code:
            body.append(html.escape(line))
            continue

        # ::: name  (open div)  /  :::  (close div)
        m = re.match(r"^:::\s*(\w+)?\s*$", line)
        if m:
            close_table(); close_list(); close_quote()
            name = m.group(1)
            if name:
                if in_div:
                    body.append('</div>')
                body.append(f'<div class="{name}">')
                in_div = name
            else:
                if in_div:
                    body.append('</div>')
                    in_div = None
            continue

        # h1/h2/h3/h4
        m = re.match(r"^(#{1,4})\s+(.+?)\s*$", line)
        if m:
            close_table(); close_list(); close_quote()
            level = len(m.group(1))
            text = m.group(2)
            slug = slugify(text)
            if level == 1:
                page_title = text
                body.append(f'<h1 id="{slug}">{inline(text)}</h1>')
            else:
                body.append(f'<h{level} id="{slug}">{inline(text)}</h{level}>')
            continue

        # Tables
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                close_list(); close_quote()
                in_table = True
                table_buf = [line]
            else:
                table_buf.append(line)
            continue
        if in_table:
            close_table()

        # Blockquotes (callouts)
        if line.startswith("> "):
            if not in_quote:
                close_list()
                body.append('<blockquote>')
                in_quote = True
            body.append(f'<p>{inline(line[2:])}</p>')
            continue
        if in_quote:
            close_quote()

        # Bullet lists
        m = re.match(r"^(\s*)-\s+(.+?)\s*$", line)
        if m:
            if not in_list:
                body.append('<ul>')
                in_list = True
            body.append(f'<li>{inline(m.group(2))}</li>')
            continue
        if in_list and not line.strip().startswith("-"):
            close_list()

        # Blank
        if not line.strip():
            continue

        # Plain paragraph
        body.append(f"<p>{inline(line)}</p>")

    close_table(); close_list(); close_quote()
    if in_div:
        body.append('</div>')
    if in_code:
        body.append('</code></pre>')

    nav_html = build_nav(headings)

    out = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{html.escape(page_title)}</title>",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<style>{CSS}</style>",
        "</head><body>",
        nav_html,
        '<main class="content">',
        *body,
        "</main>",
        "</body></html>",
    ]
    with open(sys.argv[2], "w") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
