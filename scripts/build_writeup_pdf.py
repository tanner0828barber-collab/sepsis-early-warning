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
CSS = """@page{size:Letter;margin:0.4in}
body{margin:0;font-family:Helvetica,Arial,sans-serif;font-size:8.6pt;line-height:1.27;color:#111}
h1{font-size:13pt;margin:0 0 4px}h2{font-size:10pt;margin:6px 0 1px}p{margin:2px 0}ol{margin:2px 0 2px 18px;padding:0}
table{border-collapse:collapse;margin:3px 0;font-size:8pt}td,th{border:1px solid #bbb;padding:1.5px 5px;text-align:left}code{font-size:8pt}"""

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
