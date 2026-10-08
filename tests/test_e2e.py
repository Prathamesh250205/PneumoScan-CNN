"""Playwright smoke test. Run the app first: `uvicorn main:app --port 8000`, then `pytest tests`.
Override target with BASE_URL (e.g. the Vercel URL)."""
import os
import re
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright, expect

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
OUT = Path(__file__).parent / "screenshots"


def test_landing_and_predict(tmp_path):
    xray = tmp_path / "xray.png"
    Image.open(Path(__file__).parents[1] / "public/images/gradcam_normal_example.webp").crop((0, 0, 590, 420)).save(xray)
    OUT.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, viewport in {"desktop": (1366, 900), "mobile": (390, 844)}.items():
            page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(BASE_URL)

            expect(page.locator("h1")).to_contain_text("wrong")
            expect(page.locator("#status")).to_have_attribute("data-state", re.compile("warn|ok"))
            # No horizontal scroll on any viewport
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")

            page.set_input_files("#file", str(xray))
            page.click("#analyze")
            expect(page.locator("#verdict")).to_have_attribute("data-level", re.compile("low|mid|high"), timeout=60_000)
            p_pneu = float(page.inner_text("#p-pneu"))
            assert 0 <= p_pneu <= 1

            # Bundled samples run end to end; with real weights they must be classified correctly
            page.click("[data-sample=pneumonia]")
            expect(page.locator("#verdict")).to_have_text("Pneumonia likely", timeout=60_000)
            page.click("[data-sample=normal]")
            expect(page.locator("#verdict")).to_have_text("Normal likely", timeout=60_000)

            page.screenshot(path=OUT / f"{name}.png", full_page=True)
            assert not errors, errors
        browser.close()
