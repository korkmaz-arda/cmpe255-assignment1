"""Browser E2E fixtures: a real uvicorn server on a temporary database per test."""

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, url: str):
        self.url = url

    def api(self, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else None

    def today(self) -> str:
        return self.api("GET", "/api/workspace")["today"]


@pytest.fixture
def server(tmp_path):
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "app", "--port", str(port), "--db", str(tmp_path / "e2e.db"), "--no-seed"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15
    while True:
        try:
            urllib.request.urlopen(url + "/api/workspace", timeout=1)
            break
        except OSError:
            if proc.poll() is not None or time.time() > deadline:
                proc.kill()
                raise RuntimeError(f"server failed to start: {proc.stdout.read().decode()}")
            time.sleep(0.1)
    yield Server(url)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def page_errors():
    return []


@pytest.fixture
def open_app(context, server, page_errors):
    """Open the workspace in a fresh tab; records feedback cues and JS errors."""
    def _open(install_clock: bool = False):
        page = context.new_page()
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.on("dialog", lambda dialog: dialog.accept())
        page.add_init_script(
            "window.__feedback = [];"
            "document.addEventListener('zenith:feedback', e => window.__feedback.push(e.detail.kind));")
        if install_clock:
            page.clock.install()
        page.goto(server.url)
        page.locator("#sync-status.live").wait_for()
        return page
    yield _open
    assert page_errors == [], f"JavaScript errors: {page_errors}"


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    # Desktop-sized viewport so drag targets (e.g. lower matrix quadrants) need no scrolling mid-drag.
    return {**browser_context_args, "viewport": {"width": 1440, "height": 1100}}
