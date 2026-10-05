"""Run: python app.py. A local dashboard with no remote services or telemetry."""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import threading
from urllib.parse import urlparse

from risk_engine import Settings, analyze, demo_csv, safe_csv

ROOT = Path(__file__).resolve().parent
BUSY = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def send(self, content, status=200, content_type="application/json; charset=utf-8"):
        if not isinstance(content, bytes):
            content = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(content)

    def error(self, message, status=400):
        self.send(json.dumps({"error": message}), status)

    def permitted_host(self):
        return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    def do_GET(self):
        if not self.permitted_host():
            return self.error("Use localhost or 127.0.0.1.", 403)
        route = urlparse(self.path).path
        if route == "/":
            self.send((ROOT / "dashboard.html").read_bytes(), content_type="text/html; charset=utf-8")
        elif route == "/api/demo":
            self.send(demo_csv(), content_type="text/csv; charset=utf-8")
        elif route == "/api/health":
            self.send(json.dumps({"status": "ok", "today": date.today().isoformat()}))
        else:
            self.error("Not found.", 404)

    def do_POST(self):
        if not self.permitted_host():
            return self.error("Use localhost or 127.0.0.1.", 403)
        expected_origin = "http://" + self.headers.get("Host", "")
        if self.headers.get("Origin") not in {None, expected_origin}:
            return self.error("Cross-origin requests are not allowed.", 403)
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.error("Use application/json.", 415)
        route = urlparse(self.path).path
        if route not in {"/api/analyze", "/api/export"}:
            return self.error("Not found.", 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 3_000_000:
                raise ValueError("Request must be smaller than 3 MB.")
            self.connection.settimeout(30)
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Request must be a JSON object.")
            if route == "/api/export":
                rows = body.get("rows")
                if not isinstance(rows, list) or len(rows) > 6000 or any(not isinstance(x, dict) for x in rows):
                    raise ValueError("Export requires a list of records.")
                return self.send(safe_csv(rows), content_type="text/csv; charset=utf-8")
            raw = body.get("settings", {})
            if not isinstance(raw, dict) or set(raw)-{f.name for f in fields(Settings)}:
                raise ValueError("Unknown settings.")
            settings = Settings(**raw)
            settings.validate()
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            return self.error(str(exc))
        if not BUSY.acquire(blocking=False):
            return self.error("Another simulation is running. Try again shortly.", 429)
        try:
            result = analyze(body.get("csv", ""), settings)
            self.send(json.dumps(result, allow_nan=False))
        except (ValueError, TypeError, KeyError) as exc:
            self.error(str(exc))
        except Exception:
            self.error("Analysis failed unexpectedly. Check the input and restart the application.", 500)
        finally:
            BUSY.release()

    def log_message(self, format, *args):
        # Do not log uploaded content or borrower identifiers.
        pass


def main():
    parser = argparse.ArgumentParser(description="DealerRisk: local Python credit portfolio dashboard")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port from 1024 to 65535.")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"DealerRisk is ready: http://127.0.0.1:{args.port}\nPress Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
