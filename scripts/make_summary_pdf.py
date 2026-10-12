"""Render docs/summary.html to public/project_summary.pdf (the site's "Download summary" link).

    python scripts/make_summary_pdf.py
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto((ROOT / "docs" / "summary.html").as_uri(), wait_until="networkidle")
    page.pdf(path=str(ROOT / "public" / "project_summary.pdf"), format="A4", print_background=True, prefer_css_page_size=True)
    browser.close()
print("wrote public/project_summary.pdf")
