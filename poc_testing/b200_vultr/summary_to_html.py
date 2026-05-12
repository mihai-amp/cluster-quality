#!/usr/bin/env python3
"""Convert summary.md to a single-file summary.html with light table styling.

Handles only the markdown subset emitted by aggregate_results.py:
  - h1/h2/h3 headers, paragraphs, blank lines
  - GitHub-flavored tables (header row + --- separator + body rows)
  - Fenced code blocks (```)
  - **bold** and `inline code`

Usage: summary_to_html.py <input.md> <output.html>
"""

import html
import re
import sys

CSS = """
body { font-family: -apple-system, system-ui, sans-serif; max-width: 1200px;
       margin: 2em auto; padding: 0 1em; color: #222; line-height: 1.4; }
h1 { border-bottom: 2px solid #333; padding-bottom: 0.2em; }
h2 { margin-top: 2em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }
h3 { margin-top: 1.5em; color: #555; }
table { border-collapse: collapse; margin: 1em 0; font-size: 0.9em; width: 100%; }
th, td { border: 1px solid #ccc; padding: 4px 8px; }
th { background: #f4f4f4; text-align: left; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:nth-child(even) td { background: #fafafa; }
ul { margin: 0.5em 0 1em 0; padding-left: 1.5em; }
li { margin: 0.15em 0; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px;
       font-family: ui-monospace, Menlo, monospace; font-size: 0.9em; }
pre { background: #f6f8fa; padding: 1em; overflow-x: auto;
      border-radius: 4px; font-size: 0.85em; }
pre code { background: transparent; padding: 0; }
"""

INLINE_CODE = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*([^*]+)\*\*")


def inline(text):
    text = html.escape(text)
    text = INLINE_CODE.sub(r"<code>\1</code>", text)
    text = BOLD.sub(r"<strong>\1</strong>", text)
    return text


def is_separator(row_cells):
    # GFM separator row: each cell is dashes optionally bracketed by colons
    return all(re.match(r":?-+:?$", c.strip()) for c in row_cells if c.strip())


def split_row(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def cell_class(cell, align):
    # Right-align numeric-looking cells (digits, optional minus/dot/percent)
    if align == "right":
        return ' class="num"'
    if re.match(r"-?\d", cell) or cell.endswith("%"):
        return ' class="num"'
    return ""


def aligns_from_sep(cells):
    out = []
    for c in cells:
        c = c.strip()
        if c.endswith(":") and c.startswith(":"):
            out.append("center")
        elif c.endswith(":"):
            out.append("right")
        else:
            out.append("left")
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: summary_to_html.py <input.md> <output.html>", file=sys.stderr)
        sys.exit(2)

    lines = open(sys.argv[1]).read().splitlines()
    out = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Phase 1 summary</title>",
        f"<style>{CSS}</style>",
        "</head><body>",
    ]

    i = 0
    in_code = False
    while i < len(lines):
        line = lines[i]

        # Fenced code block
        if line.startswith("```"):
            if not in_code:
                out.append("<pre><code>")
                in_code = True
            else:
                out.append("</code></pre>")
                in_code = False
            i += 1
            continue
        if in_code:
            out.append(html.escape(line))
            i += 1
            continue

        # Headers
        if line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
            i += 1
            continue
        if line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
            i += 1
            continue
        if line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
            i += 1
            continue

        # Table: header | --- | rows
        if line.lstrip().startswith("|") and i + 1 < len(lines) and is_separator(split_row(lines[i + 1])):
            header_cells = split_row(line)
            aligns = aligns_from_sep(split_row(lines[i + 1]))
            out.append("<table>")
            out.append("<thead><tr>" + "".join(
                f'<th{cell_class(c, aligns[j] if j < len(aligns) else "left")}>{inline(c)}</th>'
                for j, c in enumerate(header_cells)
            ) + "</tr></thead>")
            out.append("<tbody>")
            i += 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = split_row(lines[i])
                out.append("<tr>" + "".join(
                    f'<td{cell_class(c, aligns[j] if j < len(aligns) else "left")}>{inline(c)}</td>'
                    for j, c in enumerate(cells)
                ) + "</tr>")
                i += 1
            out.append("</tbody></table>")
            continue

        # Bullet list: consecutive lines starting with "- " or "* "
        if re.match(r"^\s*[-*]\s+", line):
            out.append("<ul>")
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                item = re.sub(r"^\s*[-*]\s+", "", lines[i])
                out.append(f"<li>{inline(item)}</li>")
                i += 1
            out.append("</ul>")
            continue

        # Blank line: paragraph break (ignore — paragraph wrapping below handles it)
        if not line.strip():
            i += 1
            continue

        # Plain paragraph line
        out.append(f"<p>{inline(line)}</p>")
        i += 1

    out.append("</body></html>")
    open(sys.argv[2], "w").write("\n".join(out))


if __name__ == "__main__":
    main()
