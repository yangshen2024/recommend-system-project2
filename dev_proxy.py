#!/usr/bin/env python3
"""Dev proxy: serves static files on 8080 and proxies /api/ to 8096."""
import http.server
import urllib.request
import urllib.error
import sys

API_TARGET = "http://127.0.0.1:8096"

class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    # Add charset=UTF-8 for text/* content types to prevent garbled Chinese
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.html': 'text/html; charset=utf-8',
        '.htm': 'text/html; charset=utf-8',
        '.css': 'text/css; charset=utf-8',
        '.js': 'application/javascript; charset=utf-8',
        '.json': 'application/json; charset=utf-8',
        '.xml': 'application/xml; charset=utf-8',
        '.svg': 'image/svg+xml; charset=utf-8',
        '.txt': 'text/plain; charset=utf-8',
    }

    def do_GET(self):
        if self.path.startswith("/api/"):
            self._proxy("GET")
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/"):
            self._proxy("POST")
        else:
            super().do_POST()

    def _proxy(self, method):
        url = API_TARGET + self.path
        data = None
        if method == "POST":
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length) if length else None

        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if body:
                self.wfile.write(body)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Proxy error: {e}".encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

if __name__ == "__main__":
    addr = ("127.0.0.1", 8080)
    httpd = http.server.HTTPServer(addr, ProxyHandler)
    print(f"Dev proxy running at http://{addr[0]}:{addr[1]}")
    print(f"/api/* → {API_TARGET}")
    httpd.serve_forever()
