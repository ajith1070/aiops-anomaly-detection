import pytest

from detector.anomaly_detector import AnomalyDetector


@pytest.fixture
def detector():
    model = AnomalyDetector()
    model.train([
        [25, 40], [28, 42], [30, 39], [32, 45], [27, 43],
        [35, 46], [29, 41], [31, 44], [26, 38], [33, 42],
        [24, 41], [34, 45], [30, 43], [28, 40], [32, 41],
        [27, 44], [29, 42], [31, 40], [26, 45], [33, 43],
    ])
    return model


def test_normal_metrics(detector):
    result = detector.predict(30, 42)
    assert result["status"] == "NORMAL"
    assert result["anomaly"] is False


def test_high_load_is_anomaly(detector):
    result = detector.predict(98, 95)
    assert result["status"] == "ANOMALY"
    assert result["anomaly"] is True


def test_prediction_requires_training():
    with pytest.raises(RuntimeError):
        AnomalyDetector().predict(30, 42)


@pytest.mark.parametrize("cpu,memory", [(-1, 50), (101, 50), (50, 101)])
def test_invalid_metrics(detector, cpu, memory):
    with pytest.raises(ValueError):
        detector.predict(cpu, memory)
