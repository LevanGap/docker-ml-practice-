from flask import Flask, request, jsonify
import joblib
from schemas import PredictRequest
import numpy as np
from db_utils import log_prediction, get_connection
from pydantic import ValidationError
import os
from model_manager import ModelManager

API_KEY = os.getenv("API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "model_v1.pkl")


app = Flask(__name__)

model_manager = ModelManager()

from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header != f"Bearer {API_KEY}":
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function


@app.route("/reload-model", methods=["POST"])
@require_api_key
def reload_model():
    try:
        data = request.get_json(force=True)
        filename = data.get("filename")
        if not filename:
            return jsonify({"error": "No filename provided"}), 400

        model_manager.load_model(filename)
        return jsonify({"status": f"Model reloaded successfully", "model_version": model_manager.model_version})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
@require_api_key
def predict():
    try:
        # Validate input
        data = request.get_json(force=True)
        validated = PredictRequest(**data)

        # Model prediction
        features = np.array(validated.features).reshape(1, -1)
        prediction = model_manager.predict(features)
        pred_value = int(prediction[0])

        # Log to DB
        log_prediction(validated.features, pred_value)

        return jsonify({
        "prediction": pred_value,
        "model_version": model_manager.model_version})


    except ValidationError as ve:
        return jsonify({"error": ve.errors()}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/logs", methods=["GET"])
def logs():
    """
    Returns the last 10 predictions from the database.
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, input_features, prediction, created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT 10
        """)
        rows = cursor.fetchall()
        logs = []
        for row in rows:
            logs.append({
                "id": row[0],
                "features": row[1],
                "prediction": row[2],
                "created_at": row[3].isoformat()
            })
        return jsonify(logs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if conn:
            conn.close()

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({"error": "Bad request"}), 400

if __name__ == "__main__":
    # Load default model at startup
    try:
        model_manager.load_model("model_v1.pkl")  # or your default model
        print(f"Model {model_manager.model_version} loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load default model: {e}")
    
    app.run(host="0.0.0.0", port=8080)