# AIOps Anomaly Detection and Monitoring

A hands-on AIOps project that uses machine learning to identify abnormal CPU and memory utilization, exposes application metrics, and visualizes monitoring data through Prometheus and Grafana.

## Project Overview

This project demonstrates how machine learning and monitoring tools can work together to detect resource-utilization anomalies.

The application uses the Isolation Forest algorithm from Scikit-learn to classify CPU and memory utilization data as normal or anomalous.

## Architecture

```text
                 Client
                   |
                   v
             Flask REST API
                   |
                   v
          Isolation Forest Model
                   |
             Prediction Result
                   |
                   v
            Prometheus Metrics
                   |
                   v
                Grafana
              Dashboard
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Flask | REST API |
| Scikit-learn | Anomaly detection |
| Isolation Forest | Machine learning algorithm |
| Docker | Containerization |
| Docker Compose | Service orchestration |
| Prometheus | Metrics collection |
| Grafana | Monitoring dashboards |
| Pytest | Automated testing |
| GitHub Actions | CI workflow |

## Features

- CPU and memory utilization anomaly detection.
- Flask REST API for submitting metrics.
- Prometheus metrics endpoint.
- Grafana dashboard with prediction monitoring.
- Dockerized application.
- Docker Compose orchestration.
- Automated tests using Pytest.
- GitHub Actions CI workflow.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Application homepage |
| GET | `/health` | Health check |
| POST | `/predict` | Detect anomalies |
| GET | `/metrics` | Prometheus metrics |

### Example prediction request

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"cpu_percent":98,"memory_percent":95}'
```

Example response:

```json
{
  "anomaly": true,
  "cpu_percent": 98.0,
  "memory_percent": 95.0,
  "status": "ANOMALY"
}
```

## Run the Project

Clone the repository:

```bash
git clone https://github.com/ajith1070/aiops-anomaly-detection.git
cd aiops-anomaly-detection
```

Start the application:

```bash
docker compose up -d --build
```

Access the services:

- Flask API: http://localhost:5000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Grafana's default local credentials are `admin` / `admin` unless changed through configuration.

## Run Tests

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python -m pytest -v
```

## CI Workflow

GitHub Actions automatically runs the test suite and Python compilation checks when changes are pushed to the repository.

Workflow file:

`.github/workflows/ci.yml`

## Monitoring

Prometheus collects application prediction metrics through `/metrics`.

Grafana provides a provisioned dashboard containing:

- Total anomaly predictions.
- Prediction counts grouped by status.

## Project Limitations

- The model currently uses sample training data.
- Predictions are demonstrated through submitted CPU and memory values.
- This is a hands-on AIOps demonstration, not a production infrastructure monitoring platform.

## Future Enhancements

- Integrate live infrastructure metrics.
- Add alerting through Alertmanager.
- Improve model training using historical monitoring data.
- Implement automated Docker image publishing and deployment.

## Author

Ajith

GitHub: https://github.com/ajith1070
