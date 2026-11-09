from flask import Flask, jsonify, request
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from detector.anomaly_detector import AnomalyDetector

app = Flask(__name__)

PREDICTIONS = Counter(
    "aiops_predictions_total",
    "Total anomaly predictions",
    ["status"],
)

detector = AnomalyDetector()
detector.train([
    [25, 40], [28, 42], [30, 39], [32, 45], [27, 43],
    [35, 46], [29, 41], [31, 44], [26, 38], [33, 42],
    [24, 41], [34, 45], [30, 43], [28, 40], [32, 41],
    [27, 44], [29, 42], [31, 40], [26, 45], [33, 43],
])


@app.get("/")
def home():
    return jsonify({"project": "AIOps Anomaly Detection API"})


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    try:
        cpu = float(data["cpu_percent"])
        memory = float(data["memory_percent"])
        result = detector.predict(cpu, memory)
    except KeyError as error:
        return jsonify({"error": f"Missing field: {error.args[0]}"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "CPU and memory must be valid numbers between 0 and 100"}), 400

    PREDICTIONS.labels(status=result["status"]).inc()
    return jsonify(result)


@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
