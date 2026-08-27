"""
Vercel Serverless Function — Analyze Proxy Endpoint (/api/analyze)
Standard Python WSGI Handler Proxying Request to Heavy AI Backend
"""

import os
import json
import urllib.request

def handler(environ, start_response):
    """
    WSGI compliant serverless handler proxying requests to AI Backend.
    Accepts (environ, start_response) WSGI signature.
    """
    try:
        method = environ.get("REQUEST_METHOD", "GET")
        if method == "OPTIONS":
            start_response("200 OK", [
                ("Access-Control-Allow-Origin", "*"),
                ("Access-Control-Allow-Methods", "POST, GET, OPTIONS"),
                ("Access-Control-Allow-Headers", "*"),
                ("Content-Length", "0")
            ])
            return [b""]

        backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8000").rstrip("/")
        endpoint = f"{backend_url}/api/analyze"

        try:
            content_length = int(environ.get("CONTENT_LENGTH", 0))
        except (ValueError, TypeError):
            content_length = 0

        body = b""
        if content_length > 0 and "wsgi.input" in environ:
            body = environ["wsgi.input"].read(content_length)

        content_type = environ.get("CONTENT_TYPE", "")

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
                status_code = resp.status
                status = f"{status_code} OK" if status_code == 200 else f"{status_code} Error"
                headers = [
                    ("Content-Type", "application/json"),
                    ("Access-Control-Allow-Origin", "*"),
                    ("Content-Length", str(len(resp_data)))
                ]
                start_response(status, headers)
                return [resp_data]

        except Exception as e:
            err_payload = {
                "status": "error",
                "error": f"Vercel Proxy Error: Unable to reach AI Backend at {backend_url}. Detail: {str(e)}"
            }
            err_bytes = json.dumps(err_payload).encode("utf-8")
            start_response("502 Bad Gateway", [
                ("Content-Type", "application/json"),
                ("Access-Control-Allow-Origin", "*"),
                ("Content-Length", str(len(err_bytes)))
            ])
            return [err_bytes]

    except Exception as fatal_err:
        err_bytes = json.dumps({"status": "error", "error": str(fatal_err)}).encode("utf-8")
        start_response("500 Internal Server Error", [
            ("Content-Type", "application/json"),
            ("Access-Control-Allow-Origin", "*"),
            ("Content-Length", str(len(err_bytes)))
        ])
        return [err_bytes]
