"""
Vercel Serverless Function — Analyze Proxy Endpoint (/api/analyze)
Lightweight WSGI Handler Proxying Request to Heavy AI Backend
"""

import os
import json
import urllib.request
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_POST(self):
        backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8000").rstrip("/")
        endpoint = f"{backend_url}/api/analyze"

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        content_type = self.headers.get("Content-Type", "")

        try:
            req = urllib.request.Request(
                endpoint,
                data=body,
                headers={
                    "Content-Type": content_type,
                    "User-Agent": "Vercel-Proxy"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(resp_data)
                return
        except Exception as e:
            err_payload = {
                "status": "error",
                "error": f"Vercel Proxy Error: Unable to reach AI Backend at {backend_url}. Detail: {str(e)}"
            }
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(err_payload).encode("utf-8"))
            return
