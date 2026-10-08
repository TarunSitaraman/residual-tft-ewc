"""Local live server for the data platform page (the page built by make_spark_dashboard.py).

Serves the page at http://localhost:8050 and rebuilds it in memory whenever a watched input changes (results files,
derived tables, download manifests, the Spark snapshot, the template). The page polls /api/status every few seconds,
shows what is running now (a Spark job and its task progress, the History Server, pipeline scripts) and reloads itself
after a rebuild. Read-only: it writes no files. Run from the repository root::

    python scripts/platform_server.py            # this PC only
    python scripts/platform_server.py --lan      # also reachable from other devices on the same network
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import socket
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import platform_data as PD  # noqa: E402

WATCH = ["results/*.json", "data/*.csv", "data/spark_ui_snapshot_dashboard.json", "data/raw/*manifest*.json",
         "docs/external_data/metadata/*.json", "docs/figures/spark/*.png", "scripts/spark_dashboard_template.html",
         "scripts/platform_catalog.py", "scripts/make_spark_dashboard.py"]
FIGURES = "docs/figures/spark"
SPARK_UI, HISTORY = "http://localhost:4040/api/v1", "http://localhost:18080/api/v1"
POLL_S = 5
TYPES = {".png": "image/png", ".pdf": "application/pdf"}


def fingerprint(root: Path) -> str:
    """Hash of the name, size and modification time of every watched file."""
    h = hashlib.sha256()
    for pattern in WATCH:
        for p in sorted(root.glob(pattern)):
            st = p.stat()
            h.update(f"{p.relative_to(root)}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def figure_path(root: Path, url_path: str) -> Path | None:
    """The file for /figures/spark/<name>.png, or None if the path leaves that folder or is not a PNG."""
    base = (root / FIGURES).resolve()
    p = (base / url_path.removeprefix("/figures/spark/")).resolve()
    return p if p.parent == base and p.suffix == ".png" and p.is_file() else None


def host_allowed(host_header: str | None, allowed: set[str]) -> bool:
    """True if the Host header names this machine (blocks DNS-rebinding requests from other sites)."""
    if not host_header:
        return False
    h = host_header.strip().lower()
    h = h[1:h.index("]")] if h.startswith("[") and "]" in h else h.rsplit(":", 1)[0] if h.count(":") == 1 else h
    return h in allowed


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Page:
    """The rendered page, rebuilt in a background thread when the inputs change."""

    def __init__(self) -> None:
        self.html, self.fp, self.built, self.error, self.building = "", "", "", "", False
        self.lock = threading.Lock()

    def refresh(self) -> None:
        fp = fingerprint(ROOT)
        with self.lock:
            if fp == self.fp or self.building:
                return
            self.building = True
        try:
            import platform_catalog
            import make_spark_dashboard as M
            importlib.reload(platform_catalog)
            M = importlib.reload(M)
            data, _ = M.build()
            data["live"] = {"fingerprint": fp, "built_utc": _now()}
            html = M.render(data)
            with self.lock:
                self.html, self.fp, self.built, self.error = html, fp, data["live"]["built_utc"], ""
            print(f"[{_now()}] page rebuilt ({len(html) / 1e3:.0f} kB, inputs {fp})", flush=True)
        except Exception:  # keep serving the last good page; the status bar shows the error
            with self.lock:
                self.error = traceback.format_exc(limit=3)
            print(self.error, file=sys.stderr, flush=True)
        finally:
            with self.lock:
                self.building = False


def _get(url: str):
    try:
        with urllib.request.urlopen(url, timeout=0.8) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def spark_status() -> dict:
    apps = _get(f"{SPARK_UI}/applications")
    if not apps:
        return {"running": False}
    app = apps[0]
    jobs = _get(f"{SPARK_UI}/applications/{app['id']}/jobs?status=running") or []
    return {"running": True, "app": app["name"], "app_id": app["id"],
            "jobs": [{"desc": j.get("description") or j["name"], "done": j["numCompletedTasks"], "total": j["numTasks"],
                      "active": j["numActiveTasks"], "failed": j["numFailedTasks"]} for j in jobs]}


def history_status() -> dict:
    apps = _get(f"{HISTORY}/applications")
    if apps is None:
        return {"up": False}
    last = apps[0] if apps else None
    return {"up": True, "apps": len(apps), "last": None if not last else {
        "name": last["name"], "id": last["id"], "completed": last["attempts"][-1]["completed"],
        "minutes": round(last["attempts"][-1].get("duration", 0) / 60000, 1)}}


def pipeline_processes() -> list[dict]:
    """Running Python or PowerShell processes started from a script in scripts/ (script name only, no arguments)."""
    known = {p.name for p in (ROOT / "scripts").iterdir() if p.suffix in (".py", ".ps1")} - {Path(__file__).name}
    out = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            args = proc.info["cmdline"] or []
        except (psutil.Error, KeyError):
            continue
        hit = next((Path(a).name for a in args if Path(a).name in known), None)
        if hit:
            out.append({"script": hit, "pid": proc.info["pid"],
                        "since": datetime.fromtimestamp(proc.info["create_time"], timezone.utc).isoformat(timespec="seconds")})
    return sorted(out, key=lambda p: p["since"])


class Status:
    def __init__(self, page: Page) -> None:
        self.page, self.value = page, {}

    def loop(self) -> None:
        while True:
            self.page.refresh()
            v = {"checked_utc": _now(), "spark": spark_status(), "history": history_status(), "processes": pipeline_processes()}
            with self.page.lock:
                v["page"] = {"fingerprint": self.page.fp, "built_utc": self.page.built, "building": self.page.building,
                             "error": self.page.error.strip().splitlines()[-1] if self.page.error else ""}
            self.value = v
            time.sleep(POLL_S)


def handler(page: Page, status: Status, allowed: set[str]):
    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if not host_allowed(self.headers.get("Host"), allowed):
                self._send(421, b"Unknown host", "text/plain; charset=utf-8")
                return
            path, _, query = self.path.partition("?")
            if path.startswith("/api/data/"):
                self._data(path, {k: v[-1] for k, v in urllib.parse.parse_qs(query).items()})
            elif path.startswith("/files/") and (f := PD.allowed(ROOT, urllib.parse.unquote(path[7:]), "files")):
                self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))
            elif path in ("/", "/index.html"):
                with page.lock:
                    html = page.html
                if not html:
                    self._send(503, b"Building the page, refresh in a few seconds.", "text/plain; charset=utf-8")
                else:
                    self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/status":
                self._send(200, json.dumps(status.value).encode(), "application/json")
            elif path.startswith("/figures/spark/") and (f := figure_path(ROOT, path)):
                self._send(200, f.read_bytes(), "image/png")
            else:
                self._send(404, b"Not found", "text/plain; charset=utf-8")

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, allow_nan=False, default=str).encode(), "application/json")

        def _data(self, path: str, q: dict) -> None:
            """Read-only data API; every file is looked up in the platform_data catalogue."""
            try:
                kind = path.removeprefix("/api/data/")
                if kind == "catalog":
                    return self._json(PD.catalog(ROOT))
                if kind in ("table", "stats", "json"):
                    f = PD.allowed(ROOT, q.get("path", ""), "results" if kind == "json" else "tables")
                    if f is None:
                        return self._json({"error": "unknown file"}, 404)
                    if kind == "json":
                        return self._send(200, f.read_bytes(), "application/json")
                    if kind == "stats":
                        return self._json(PD.table_stats(f))
                    return self._json(PD.table_page(f, int(q.get("offset", 0)), int(q.get("limit", 100)), q.get("sort") or None,
                                                    q.get("desc") == "1", q.get("column") or None, q.get("q", "")))
                if kind in ("raw-files", "raw-preview"):
                    ds = q.get("dataset", "")
                    if ds not in PD.RAW_MANIFESTS:
                        return self._json({"error": "unknown dataset"}, 404)
                    if kind == "raw-files":
                        return self._json(PD.raw_files(ROOT, ds))
                    return self._json(PD.raw_preview(ROOT, ds, q.get("path", "")))
                return self._json({"error": "unknown endpoint"}, 404)
            except (KeyError, FileNotFoundError) as e:
                return self._json({"error": f"not found: {e}"}, 404)
            except ValueError as e:
                return self._json({"error": f"bad request: {e}"}, 400)

        def log_message(self, *args) -> None:  # quiet: the page polls every few seconds
            pass
    return H


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--lan", action="store_true", help="listen on all interfaces, not only this PC")
    args = ap.parse_args()
    page = Page()
    status = Status(page)
    threading.Thread(target=status.loop, daemon=True).start()
    allowed = {"localhost", "127.0.0.1", "::1"} | ({lan_ip(), socket.gethostname().lower()} if args.lan else set())
    server = ThreadingHTTPServer(("0.0.0.0" if args.lan else "127.0.0.1", args.port), handler(page, status, allowed))
    print(f"Data platform page: http://localhost:{args.port}"
          + (f"  (network: http://{lan_ip()}:{args.port})" if args.lan else ""), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
