"""Build report.pdf from report.md.

    pip install markdown
    python build_report.py

Converts the Markdown to HTML and prints it to PDF with headless Microsoft Edge or Chrome.
Run it again after every edit to report.md (e.g. after filling in the group names).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium", "chromium-browser", "chrome",
]

CSS = """
@page { size: A4; margin: 15mm 16mm; }
body { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 10pt; line-height: 1.38;
       color: #111; }
h1.title { font-size: 17pt; text-align: center; margin: 0 0 2pt; }
p.subtitle { text-align: center; margin: 0 0 10pt; }
h1 { font-size: 13pt; margin: 12pt 0 4pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; }
p { margin: 4pt 0; }
table { border-collapse: collapse; margin: 6pt auto; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2pt 6pt; }
th { background: #f0f0f0; }
code { font-family: Consolas, monospace; font-size: 9pt; }
figure { margin: 6pt 0; page-break-inside: avoid; }
figure img { width: 100%; }
figcaption { font-size: 9pt; color: #444; text-align: center; }
"""


def to_html(text):
    title = subtitle = ""
    front = re.match(r"^---\n(.*?)\n---\n", text, re.S)          # pandoc-style title block
    if front:
        text = text[front.end():]
        title = (re.search(r'title:\s*"(.*)"', front.group(1)) or [None, ""])[1]
        subtitle = (re.search(r'subtitle:\s*"(.*)"', front.group(1)) or [None, ""])[1]

    # ![caption](path){width=100%}  ->  figure with caption
    def figure(m):
        return f'<figure><img src="{m.group(2)}"><figcaption>{m.group(1)}</figcaption></figure>'
    text = re.sub(r"!\[(.*?)\]\((.*?)\)(\{.*?\})?", figure, text)

    body = markdown.markdown(text, extensions=["tables"])
    head = f'<h1 class="title">{title}</h1><p class="subtitle">{markdown.markdown(subtitle)[3:-4]}</p>'
    return f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>' \
           f"<style>{CSS}</style></head><body>{head}{body}</body></html>"


def main():
    html_path = HERE / "report.html"
    html_path.write_text(to_html((HERE / "report.md").read_text(encoding="utf-8")), encoding="utf-8")
    browser = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if browser is None:
        sys.exit("No Edge/Chrome found: open report.html in a browser and print it to PDF.")
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={HERE / 'report.pdf'}", html_path.as_uri()],
                   check=True, capture_output=True)
    html_path.unlink()
    print("wrote", HERE / "report.pdf")


if __name__ == "__main__":
    main()
