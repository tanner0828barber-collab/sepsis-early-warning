"""Turn WRITEUP.md into WRITEUP.pdf and check that it is exactly one page.

Run it:  .venv/bin/python scripts/build_writeup_pdf.py

It converts the markdown to HTML with mistune, then prints it to PDF with
Google Chrome in headless mode (macOS path below). It stops with an error if
the PDF is not one Letter page, so the one page rule cannot slip by.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """@page{size:Letter;margin:0.42in 0.5in 0.4in}
body{margin:0;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;font-size:8.3pt;line-height:1.36;color:#1f262d}
.cols{column-count:2;column-gap:0.34in}
a{color:#0b5a7a;text-decoration:none}strong{color:#0d1114}
h1{font-size:2.3em;line-height:1.05;margin:0;color:#0b2f42;letter-spacing:-0.4px}
h1+p{margin:4px 0 0;font-size:1.35em;color:#0b2f42}h1+p strong{font-weight:500;color:#0b2f42}
h1+p+p{margin:3px 0 8px;padding-bottom:6px;border-bottom:1.5px solid #0b5a7a;color:#5a6672;font-size:0.96em}
h2{font-size:0.96em;margin:0 0 1px;padding-top:0.7em;color:#0b5a7a;text-transform:uppercase;letter-spacing:0.9px;break-after:avoid}
p{margin:0 0 0.4em;break-inside:avoid}ul{margin:1px 0 0.4em 1.3em;padding:0}li{margin:0 0 0.2em;break-inside:avoid}
table{border-collapse:collapse;margin:3px 0 0.7em;font-size:0.92em;width:100%;break-inside:avoid}
th{background:#edf3f6;color:#0b2f42;border-bottom:1px solid #0b5a7a;padding:0.35em 0.7em;text-align:left;vertical-align:bottom}
td{border-bottom:0.5px solid #cdd6dc;padding:0.35em 0.7em;text-align:left;vertical-align:top}
td:first-child{white-space:nowrap}
body>table:first-of-type{border-collapse:separate;border-spacing:8px 0;margin:0 -8px 4px;width:calc(100% + 16px);table-layout:fixed}
body>table:first-of-type th{background:#edf3f6;border:0;border-top:3px solid #0b5a7a;padding:0.4em 0.55em 0;font-size:2em;letter-spacing:-0.3px;white-space:nowrap}
body>table:first-of-type td{background:#edf3f6;border:0;padding:0.2em 1.1em 0.8em;line-height:1.3;color:#3d4853;white-space:normal}
.cols h2:nth-of-type(3){break-before:column}.cols>h2:first-child,.cols h2:nth-of-type(3){padding-top:0.5em}
code{font-size:0.92em}"""

html = mistune.create_markdown(plugins=["table"])((ROOT / "WRITEUP.md").read_text())
# Two-column layout: the header and number tiles run full width, then the
# sections before Results in two columns, the Results table full width, and
# the rest in two columns.
first = html.index("<h2")
results = re.search(r"<h2>[^<]*</h2>\s*<table", html).start()
results_end = html.index("</table>", results) + len("</table>")
html = (html[:first] + '<div class="cols">' + html[first:results] + "</div>" + html[results:results_end]
        + '<div class="cols">' + html[results_end:] + "</div>")
page = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
with tempfile.TemporaryDirectory() as folder:
    source = Path(folder) / "writeup.html"
    source.write_text(page)
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={ROOT / 'WRITEUP.pdf'}", f"file://{source}"],
        check=True, capture_output=True,
    )
pages = len(re.findall(rb"/Type\s*/Page[^s]", (ROOT / "WRITEUP.pdf").read_bytes()))
print("WRITEUP.pdf pages:", pages)
if pages != 1:
    sys.exit("The write-up must be exactly one page. Shorten WRITEUP.md and run this again.")
