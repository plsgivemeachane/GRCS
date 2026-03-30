"""
CEPS Phase 1: Judge UI

A lightweight Flask app that renders generated UI completions in iframes
and lets a human label each as Positive (P) or Negative (N).
"""

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("judge.html")


@app.route("/api/files")
def api_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(f.name for f in DATA_DIR.glob("*.jsonl"))
    return jsonify(files)


@app.route("/api/samples")
def api_samples():
    filename = request.args.get("file", "")
    path = DATA_DIR / filename
    if not path.exists() or not filename.endswith(".jsonl"):
        return jsonify([])
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return jsonify(samples)


@app.route("/api/label", methods=["POST"])
def api_label():
    body = request.get_json()
    filename = body.get("file", "")
    sample_id = body.get("id")
    label = body.get("label")  # "P", "N", or null

    path = DATA_DIR / filename
    if not path.exists() or not filename.endswith(".jsonl"):
        return jsonify({"error": "File not found"}), 404

    # Read all, update the target, write back
    samples = []
    updated = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s["id"] == sample_id:
                s["label"] = label
                updated = True
            samples.append(s)

    if not updated:
        return jsonify({"error": "Sample not found"}), 404

    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return jsonify({"ok": True})


def run_judge(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    """Start the judge UI server."""
    print(f"[JUDGE] Starting CEPS Judge at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
