"""
Vercel Serverless Function — Health Check Endpoint (/api/health)
Lightweight WSGI Handler (No PyTorch, No Open3D, No Streamlit)
"""

import os
import json
import urllib.request
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8000").rstrip("/")
        backend_online = False
        backend_info = {}

        try:
            req = urllib.request.Request(f"{backend_url}/api/health", headers={"User-Agent": "Vercel-Health-Check"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    backend_online = True
                    backend_info = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            backend_info = {"error": str(e)}

        response_payload = {
            "status": "ok",
            "service": "DepthWizard Vercel Serverless Function",
            "backend_url": backend_url,
            "backend_online": backend_online,
            "backend_details": backend_info
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(response_payload).encode("utf-8"))
        return
