"""Headless-Chromium check of the static site (spec §7).

Serves site/ locally, opens every section at 390 px and 1280 px in light and dark mode,
exercises the four interactions, fails on console errors, and saves screenshots to
results/screenshots/.

    python scripts/screenshot_site.py
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

from config import ROOT

SECTIONS = ["question", "data", "preprocessing", "methods", "evaluation", "results",
            "inspector", "errors", "limitations", "conclusion", "reproduce"]
OUT = ROOT / "results" / "screenshots"


def serve() -> tuple[http.server.ThreadingHTTPServer, int]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the console clean
            pass

    handler = functools.partial(Quiet, directory=str(ROOT / "site"))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    server, port = serve()
    base = f"http://127.0.0.1:{port}/index.html"
    errors: list[str] = []
    checks: dict[str, object] = {}
    with sync_playwright() as p:
        # use a pre-installed Chromium if present (no download), else Playwright's own
        exe = os.environ.get("CHROMIUM_PATH", "/opt/pw-browsers/chromium")
        browser = p.chromium.launch(executable_path=exe if Path(exe).exists() else None)
        for width, height in ((1280, 900), (390, 844)):
            for scheme in ("light", "dark"):
                ctx = browser.new_context(viewport={"width": width, "height": height}, color_scheme=scheme)
                page = ctx.new_page()
                tag = f"{width}-{scheme}"
                page.on("console", lambda m, tag=tag: errors.append(f"[{tag}] {m.type}: {m.text}")
                        if m.type in ("error", "warning") else None)
                page.on("pageerror", lambda e, tag=tag: errors.append(f"[{tag}] pageerror: {e}"))
                page.goto(base)
                page.wait_for_function("document.querySelector('#t-test tbody') !== null")
                for s in SECTIONS:
                    page.goto(f"{base}#{s}")
                    page.wait_for_timeout(350)
                    if s == "inspector":
                        page.click("#col-model .prod >> nth=0")
                        page.wait_for_timeout(200)
                    page.screenshot(path=str(OUT / f"{s}-{tag}.png"), full_page=s != "inspector")
                    scroll_w = page.evaluate("document.documentElement.scrollWidth")
                    if scroll_w > width + 1:
                        errors.append(f"[{tag}] horizontal scroll on #{s}: {scroll_w}px")
                if width == 1280 and scheme == "light":
                    # interaction checks
                    page.goto(f"{base}#inspector")
                    page.fill("#q-search", page.evaluate("I.queries[5].q.slice(0, 6)"))
                    page.wait_for_selector("#q-suggest li")
                    checks["autocomplete_suggestions"] = page.locator("#q-suggest li").count()
                    page.keyboard.press("Enter")
                    checks["links_drawn"] = page.locator("#links line").count()
                    page.click("#col-bm25 .prod >> nth=1")
                    checks["contribution_bars"] = page.locator(".bar-row").count()
                    page.click("#chips .fchip >> nth=0")
                    checks["category_filter_count"] = page.inner_text("#qlist-count")
                    page.goto(f"{base}#errors")
                    page.wait_for_timeout(300)
                    box = page.locator("#c-cats").bounding_box()
                    page.mouse.click(box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.12)
                    page.wait_for_timeout(300)
                    checks["category_click_opens"] = page.evaluate("location.hash")
                    page.goto(f"{base}#results")
                    page.wait_for_timeout(300)
                    checks["delta_click_opens"] = page.evaluate(
                        "(() => { const c = Chart.getChart('c-delta'); const m = c.getDatasetMeta(0);"
                        " const i = c.data.datasets[0].data.indexOf(Math.max(...c.data.datasets[0].data));"
                        " return [m.data[i].x, m.data[i].y]; })()")
                    x, y = checks["delta_click_opens"]
                    cb = page.locator("#c-delta").bounding_box()
                    page.mouse.click(cb["x"] + x, cb["y"] + y + 4)
                    page.wait_for_timeout(300)
                    checks["delta_click_opens"] = [page.evaluate("location.hash"), page.inner_text("#active-filter")]
                    page.goto(f"{base}#evaluation")
                    page.wait_for_timeout(300)
                    checks["ablation_chart"] = page.evaluate("!!Chart.getChart('c-ablation')")
                if width == 390 and scheme == "light":
                    page.goto(f"{base}#question")
                    page.click("#menu-btn")
                    page.wait_for_timeout(300)
                    page.screenshot(path=str(OUT / "menu-390-light.png"))
                    page.click("#nav a >> nth=6")
                    page.wait_for_timeout(300)
                    checks["mobile_menu_navigates"] = page.evaluate("location.hash")
                ctx.close()
        browser.close()
    server.shutdown()
    report = {"console_errors": errors, "interaction_checks": checks}
    (OUT / "check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(f"{len(errors)} console errors / layout problems")


if __name__ == "__main__":
    main()
