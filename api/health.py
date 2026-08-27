"""
Vercel Serverless Function — Health Check Endpoint (/api/health)
Standard Python WSGI Handler (No PyTorch, No Open3D, No Streamlit)
"""

import os
import json
import urllib.request

def handler(environ, start_response):
    """
    WSGI compliant serverless handler for Vercel (@vercel/python).
    Accepts (environ, start_response) WSGI signature.
    """
    try:
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

        payload = {
            "status": "ok",
            "service": "DepthWizard Vercel Serverless Function",
            "backend_url": backend_url,
            "backend_online": backend_online,
            "backend_details": backend_info
        }
        body = json.dumps(payload).encode("utf-8")
        status = "200 OK"
        headers = [
            ("Content-Type", "application/json"),
            ("Access-Control-Allow-Origin", "*"),
            ("Content-Length", str(len(body)))
        ]
        start_response(status, headers)
        return [body]

    except Exception as err:
        err_body = json.dumps({"status": "error", "error": str(err)}).encode("utf-8")
        start_response("500 Internal Server Error", [
            ("Content-Type", "application/json"),
            ("Access-Control-Allow-Origin", "*"),
            ("Content-Length", str(len(err_body)))
        ])
        return [err_body]
