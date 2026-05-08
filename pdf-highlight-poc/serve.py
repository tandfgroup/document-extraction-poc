"""Static file server with /api/* reverse proxy to localhost:9000.

Usage:
    python3 serve.py [port]

Serves the current directory and forwards any /api/... request to the
metadata-extraction-svc port-forward on localhost:9000.
"""
import http.server
import socketserver
import sys
import urllib.parse
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
UPSTREAM = "http://localhost:9000"


class Handler(http.server.SimpleHTTPRequestHandler):
    def _proxy(self):
        url = UPSTREAM + self.path
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(url, data=body, method=self.command)
        for h, v in self.headers.items():
            if h.lower() in ("host", "content-length", "connection"):
                continue
            req.add_header(h, v)
        try:
            resp = urllib.request.urlopen(req, timeout=180)
        except urllib.error.HTTPError as e:
            resp = e
        # Stream the response (handles SSE chunked output)
        self.send_response(resp.status)
        for h, v in resp.headers.items():
            if h.lower() in ("transfer-encoding", "content-encoding", "connection"):
                continue
            self.send_header(h, v)
        self.end_headers()
        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._proxy()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            self._proxy()
        else:
            self.send_error(405)


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


with ThreadingServer(("", PORT), Handler) as httpd:
    print(f"Serving + proxying on http://localhost:{PORT}")
    print(f"  /api/* -> {UPSTREAM}")
    httpd.serve_forever()
