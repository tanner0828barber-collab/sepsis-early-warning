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
CSS = """@page{size:Letter;margin:0.4in 0.45in}
body{margin:0;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;font-size:8.5pt;line-height:1.27;color:#1a1f24}
a{color:#0b4f6c;text-decoration:none}strong{color:#0d1114}
h1{font-size:14.5pt;line-height:1.15;margin:0 0 3px;color:#0b2f42;letter-spacing:-0.1px}
h1+p{margin:0 0 5px;padding-bottom:4px;border-bottom:1.5px solid #0b4f6c;color:#4a5560}
h1+p+p{margin:0 0 3px;padding:3px 8px;background:#eef4f7;border-left:3px solid #0b4f6c;font-size:8.8pt}
h2{font-size:8.4pt;margin:5px 0 0;color:#0b4f6c;text-transform:uppercase;letter-spacing:0.7px}
p{margin:2px 0}ol{margin:2px 0 2px 18px;padding:0}
table{border-collapse:collapse;margin:3px 0;font-size:7.8pt;width:100%}
th{background:#eef4f7;color:#0b2f42;border-bottom:1px solid #0b4f6c;padding:2px 6px;text-align:left}
td{border-bottom:0.5px solid #c9d3da;padding:1.5px 6px;text-align:left}td:first-child,th:first-child{white-space:nowrap}code{font-size:7.8pt}"""

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
