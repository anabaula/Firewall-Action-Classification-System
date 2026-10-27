from pathlib import Path
import math
from flask import Flask, jsonify, render_template, request
from forest_model import ACTIONS, FEATURE_NAMES, load_model, predict

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "firewall_random_forest.json"

ACTION_DETAILS = {
    "Allow": {"description": "The model predicts this record resembles firewall traffic labeled Allow in the training data.", "explanation": "This is a prediction from historical records, not a live firewall rule or a security guarantee."},
    "Deny": {"description": "The model predicts this record resembles firewall traffic labeled Deny in the training data.", "explanation": "This is a prediction from historical records, not a live firewall rule or a security guarantee."},
    "Drop": {"description": "The model predicts this record resembles firewall traffic labeled Drop in the training data.", "explanation": "This is a prediction from historical records, not a live firewall rule or a security guarantee."},
    "Reset": {"description": "The model predicts this record resembles traffic labeled Reset-Both in the training data.", "explanation": "Reset-Both is displayed as Reset. This prediction does not control a real connection."},
}
app = Flask(__name__)


def load_saved_model():
    return load_model(MODEL_PATH) if MODEL_PATH.exists() else None


@app.get("/")
def index():
    return render_template("index.html", model_ready=MODEL_PATH.exists())


@app.get("/api/health")
def health():
    ready = MODEL_PATH.exists()
    model = load_saved_model() if ready else None
    return jsonify({"status": "ready" if ready else "model_missing", "model_ready": ready,
                    "dataset": (model or {}).get("dataset_name", "UCI Internet Firewall Data")})


@app.post("/api/predict")
def predict_route():
    model = load_saved_model()
    if model is None:
        return jsonify({"error": "The real-dataset model has not been trained yet. Provide data/log2_cleaned.csv or data/log2.csv and run: python train_model.py"}), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Could not read the submitted form. Please try again."}), 400
    try:
        values = {}
        integer_fields = {"source_port", "destination_port", "nat_source_port", "nat_destination_port", "total_packets", "forward_packets", "backward_packets"}
        for key in FEATURE_NAMES:
            value = float(payload[key])
            if not math.isfinite(value):
                raise ValueError(f"{key.replace('_', ' ').title()} must be a valid number.")
            if value < 0:
                raise ValueError("Traffic values cannot be negative.")
            if key in integer_fields and not value.is_integer():
                raise ValueError(f"{key.replace('_', ' ').title()} must be a whole number.")
            values[key] = int(value) if key in integer_fields else value
        for key in ("source_port", "destination_port", "nat_source_port", "nat_destination_port"):
            if not 0 <= values[key] <= 65535:
                raise ValueError("Ports must be between 0 and 65535.")
        if values["total_packets"] < 1:
            raise ValueError("Total packets must be at least 1.")
        if values["forward_packets"] + values["backward_packets"] > values["total_packets"]:
            raise ValueError("Forward plus backward packet counts cannot exceed total packets.")
        if values["bytes_sent"] + values["bytes_received"] > values["total_bytes"] * 1.02:
            raise ValueError("Bytes Sent plus Bytes Received should not exceed Total Bytes. Check your values.")

        feature_row = [float(values[name]) for name in FEATURE_NAMES]
        action, confidence, _probabilities = predict(model, feature_row)
        details = ACTION_DETAILS.get(action, ACTION_DETAILS["Allow"])
        return jsonify({
            "action": action,
            "confidence": confidence,
            "description": details["description"],
            "explanation": details["explanation"],
            "model_type": "Random Forest Classifier (pure Python)",
            "dataset": model.get("dataset_name", "UCI Internet Firewall Data"),
            "is_synthetic_demo_model": False,
        })
    except KeyError as exc:
        return jsonify({"error": f"Missing required input: {exc.args[0]}"}), 400
    except (TypeError, ValueError, OverflowError) as exc:
        return jsonify({"error": str(exc) or "Please check your inputs."}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
