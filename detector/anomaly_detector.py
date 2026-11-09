import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
        )
        self.trained = False

    def train(self, normal_metrics):
        data = np.asarray(normal_metrics, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2 or len(data) < 5:
            raise ValueError("Provide at least 5 CPU/memory metric pairs.")
        if not np.isfinite(data).all():
            raise ValueError("Metrics must contain finite numbers.")

        self.model.fit(data)
        self.trained = True

    def predict(self, cpu_percent, memory_percent):
        if not self.trained:
            raise RuntimeError("Train the detector before predicting.")

        values = np.asarray([[cpu_percent, memory_percent]], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Metrics must be finite.")
        if ((values < 0) | (values > 100)).any():
            raise ValueError("CPU and memory must be between 0 and 100.")

        result = self.model.predict(values)[0]
        return {
            "cpu_percent": float(cpu_percent),
            "memory_percent": float(memory_percent),
            "anomaly": bool(result == -1),
            "status": "ANOMALY" if result == -1 else "NORMAL",
        }


if __name__ == "__main__":
    normal_data = [
        [25, 40], [28, 42], [30, 39], [32, 45], [27, 43],
        [35, 46], [29, 41], [31, 44], [26, 38], [33, 42],
        [24, 41], [34, 45], [30, 43], [28, 40], [32, 41],
        [27, 44], [29, 42], [31, 40], [26, 45], [33, 43],
    ]

    detector = AnomalyDetector()
    detector.train(normal_data)

    print("Normal sample:", detector.predict(30, 42))
    print("High-load sample:", detector.predict(98, 95))
