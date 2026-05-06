from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

PREDICTION_COUNTER = Counter(
    "ml_predictions_total",
    "Total number of model predictions",
    labelnames=["class_name", "status"],
    registry=REGISTRY,
)

PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Inference latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

PREDICTION_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Distribution of predict_proba for the chosen class",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=REGISTRY,
)

ERROR_COUNTER = Counter(
    "ml_errors_total",
    "Total number of errored requests",
    labelnames=["error_type"],
    registry=REGISTRY,
)

MODEL_LOADED = Gauge(
    "ml_model_loaded",
    "Whether the model is successfully loaded (1 = yes, 0 = no)",
    registry=REGISTRY,
)

DRIFT_CHECKS = Counter(
    "ml_drift_checks_total",
    "Total number of drift checks performed",
    registry=REGISTRY,
)

DRIFT_DETECTED = Counter(
    "ml_drift_detected_total",
    "Total number of drift detections per feature",
    labelnames=["feature"],
    registry=REGISTRY,
)
