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
CSS = """@page{size:Letter;margin:0.5in 0.55in 0.45in}
body{margin:0;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;font-size:10.8pt;line-height:1.42;color:#1f262d;column-count:2;column-gap:0.34in}
a{color:#0b5a7a;text-decoration:none}strong{color:#0d1114}
h1{column-span:all;font-size:2.6em;line-height:1.05;margin:0;color:#0b2f42;letter-spacing:-0.4px}
h1+p{column-span:all;margin:4px 0 0;font-size:1.35em;color:#0b2f42}h1+p strong{font-weight:500;color:#0b2f42}
h1+p+p{column-span:all;margin:4px 0 10px;padding-bottom:8px;border-bottom:1.5px solid #0b5a7a;color:#5a6672;font-size:0.96em}
h2{font-size:0.96em;margin:0 0 2px;padding-top:0.95em;color:#0b5a7a;text-transform:uppercase;letter-spacing:0.9px;break-after:avoid}
p{margin:0 0 0.5em;break-inside:avoid}ul{margin:2px 0 0.5em 1.4em;padding:0}li{margin:0 0 0.28em;break-inside:avoid}
table{border-collapse:collapse;margin:3px 0 0.7em;font-size:0.92em;width:100%;break-inside:avoid}
th{background:#edf3f6;color:#0b2f42;border-bottom:1px solid #0b5a7a;padding:0.35em 0.7em;text-align:left;vertical-align:bottom}
td{border-bottom:0.5px solid #cdd6dc;padding:0.35em 0.7em;text-align:left;vertical-align:top}
td:first-child{white-space:nowrap}
body>table:first-of-type{column-span:all;border-collapse:separate;border-spacing:8px 0;margin:0 -8px 4px;width:calc(100% + 16px);table-layout:fixed}
body>table:first-of-type th{background:#edf3f6;border:0;border-top:3px solid #0b5a7a;padding:0.4em 0.55em 0;font-size:2em;letter-spacing:-0.3px;white-space:nowrap}
body>table:first-of-type td{background:#edf3f6;border:0;padding:0.2em 1.1em 0.8em;line-height:1.3;color:#3d4853;white-space:normal}
h2:nth-of-type(4){break-before:column}h2:has(+table){column-span:all;padding-top:1.1em}h2+table{column-span:all}
code{font-size:0.92em}"""

html = mistune.create_markdown(plugins=["table"])((ROOT / "WRITEUP.md").read_text())
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
