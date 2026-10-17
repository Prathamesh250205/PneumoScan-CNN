"""Playwright end-to-end tests. Run the app first: `uvicorn main:app --port 8000`, then `pytest tests`.
Override the target with BASE_URL (e.g. the Vercel URL)."""
import os
import re
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
OUT = Path(__file__).parent / "screenshots"
VIEWPORTS = {"desktop": (1366, 900), "mobile": (390, 844)}


def no_overflow(page):
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")


def test_landing():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, (w, h) in VIEWPORTS.items():
            page = browser.new_page(viewport={"width": w, "height": h}, color_scheme="light")
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(BASE_URL)

            expect(page.locator("h1")).to_contain_text("wrong")
            expect(page.locator("#stage")).to_have_class(re.compile("exploded"))
            no_overflow(page)

            # Theme toggle flips the theme and survives a reload
            bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
            page.click("#theme-toggle")
            expect(page.locator("html")).to_have_attribute("data-theme", "dark")
            assert page.evaluate("getComputedStyle(document.body).backgroundColor") != bg
            page.reload()
            expect(page.locator("html")).to_have_attribute("data-theme", "dark")
            page.click("#theme-toggle")
            expect(page.locator("html")).to_have_attribute("data-theme", "light")

            # WebGL attention landscape renders and the model toggle works
            page.locator("#land").scroll_into_view_if_needed()
            expect(page.locator("#land")).to_have_class(re.compile("ready"), timeout=20_000)
            page.click("[data-model=mobilenet_v2]")
            expect(page.locator("[data-model=mobilenet_v2]")).to_have_attribute("aria-pressed", "true")

            page.screenshot(path=OUT / f"landing_{name}.png", full_page=True)
            assert not errors, errors
        browser.close()


def test_research_and_demo():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, (w, h) in VIEWPORTS.items():
            page = browser.new_page(viewport={"width": w, "height": h})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{BASE_URL}/research.html")

            expect(page.locator("#status")).to_have_attribute("data-state", re.compile("warn|ok"))
            no_overflow(page)
            page.locator(".dotplot").scroll_into_view_if_needed()
            expect(page.locator(".dotplot")).to_have_class(re.compile(r"\bin\b"))

            # Bundled samples run end to end; with real weights they must be classified correctly
            page.click("[data-sample=pneumonia]")
            expect(page.locator("#verdict")).to_have_text("Pneumonia likely", timeout=60_000)
            page.click("[data-sample=normal]")
            expect(page.locator("#verdict")).to_have_text("Normal likely", timeout=60_000)
            for model in ("resnet18", "mobilenet_v2"):
                expect(page.locator(f"#cam-{model}")).to_have_attribute("src", re.compile("^data:image/jpeg"))
                assert page.evaluate(f"document.getElementById('cam-{model}').naturalWidth") > 0

            page.screenshot(path=OUT / f"research_{name}.png", full_page=True)
            assert not errors, errors
        browser.close()


def test_reduced_motion():
    """With reduced motion the pages skip animations but show the final state, with no JS errors."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(reduced_motion="reduce")
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(BASE_URL)
        expect(page.locator("#stage")).to_have_class(re.compile("exploded"))
        assert page.evaluate("getComputedStyle(document.querySelector('.stack')).animationName") == "none"
        page.goto(f"{BASE_URL}/research.html")
        expect(page.locator(".dotplot")).to_have_class(re.compile(r"\bin\b"))
        assert page.evaluate("getComputedStyle(document.querySelector('#findings .prose')).opacity") == "1"
        assert not errors, errors
        browser.close()
