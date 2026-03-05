#!/usr/bin/env python3
"""kubemark-ai Dashboard Server

Serves the static dashboard and provides a results API endpoint.

Usage:
    python3 dashboard/server.py [--port 8080] [--results-dir results/]
"""
import http.server
import json
import os
import sys
import glob
import argparse
from pathlib import Path


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Serves dashboard static files and results API."""

    results_dir = "results"

    def do_GET(self):
        if self.path == "/api/results":
            self.serve_results()
        elif self.path == "/":
            self.path = "/index.html"
            self.serve_static()
        else:
            self.serve_static()

    def serve_results(self):
        """Serve all benchmark results as JSON array."""
        results = []
        pattern = os.path.join(self.results_dir, "*/summary.json")
        for filepath in sorted(glob.glob(pattern)):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    results.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(results, indent=2).encode())

    def serve_static(self):
        """Serve static files from the dashboard directory."""
        # Resolve file path relative to dashboard directory
        dashboard_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(dashboard_dir, self.path.lstrip("/"))

        if not os.path.isfile(filepath):
            self.send_error(404, f"File not found: {self.path}")
            return

        # Content types
        ext = os.path.splitext(filepath)[1]
        content_types = {
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".png": "image/png",
            ".svg": "image/svg+xml",
        }

        self.send_response(200)
        self.send_header("Content-Type", content_types.get(ext, "application/octet-stream"))
        self.end_headers()
        with open(filepath, "rb") as f:
            self.wfile.write(f.read())

    def log_message(self, format, *args):
        """Custom log format."""
        print(f"[dashboard] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="kubemark-ai Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--results-dir", default="results", help="Directory containing benchmark results")
    args = parser.parse_args()

    DashboardHandler.results_dir = args.results_dir

    server = http.server.HTTPServer(("", args.port), DashboardHandler)
    print(f"kubemark-ai Dashboard")
    print(f"  URL:     http://localhost:{args.port}")
    print(f"  Results: {os.path.abspath(args.results_dir)}/")
    print(f"  Press Ctrl+C to stop")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
