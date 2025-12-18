from flask import Flask, jsonify, request

import os
import sys

# Add parent directory to path so imports work from api/ subfolder
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from recommendation.recommender import Recommender

CATALOG_PATH = "data/shl_catalog.json"
RULES_PATH = "config/rules.yaml"
MODEL_PATH = "models/ranking_model.pkl"

app = Flask(__name__)
recommender = Recommender(
    catalog_path=CATALOG_PATH,
    rules_path=RULES_PATH,
    model_path=MODEL_PATH,
    top_n=5,
)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/recommend", methods=["POST"])
def recommend():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "Missing JSON payload"}), 400

    recommendations = recommender.recommend(payload)
    return jsonify(recommendations)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
