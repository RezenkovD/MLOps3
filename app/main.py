import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .drift import DriftDetector
from .logging_config import setup_logging
from .metrics import (
    DRIFT_CHECKS,
    DRIFT_DETECTED,
    ERROR_COUNTER,
    MODEL_LOADED,
    PREDICTION_CONFIDENCE,
    PREDICTION_COUNTER,
    PREDICTION_LATENCY,
    REGISTRY,
)
from .schemas import (
    DriftRequest,
    DriftResponse,
    IrisFeatures,
    PredictionResponse,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
REFERENCE_PATH = Path(__file__).resolve().parent.parent / "reference_stats.joblib"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

setup_logging()
logger = logging.getLogger("ml-api")

state = {"model": None, "drift_detector": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        MODEL_LOADED.set(0)
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    state["model"] = joblib.load(MODEL_PATH)
    MODEL_LOADED.set(1)

    if REFERENCE_PATH.exists():
        ref_data = joblib.load(REFERENCE_PATH)
        state["drift_detector"] = DriftDetector(
            reference=ref_data["X"],
            feature_names=ref_data["feature_names"],
        )
        logger.info(
            "startup_complete",
            extra={"event": "startup", "model_loaded": True, "drift_detector_ready": True},
        )
    else:
        logger.warning(
            "reference_missing",
            extra={"event": "startup", "model_loaded": True, "drift_detector_ready": False},
        )

    yield

    state["model"] = None
    state["drift_detector"] = None
    MODEL_LOADED.set(0)


app = FastAPI(
    title="Iris ML API with Monitoring",
    description="ML API з Prometheus-метриками та drift detection",
    version="3.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    if request.url.path == "/predict":
        PREDICTION_LATENCY.observe(elapsed)
    return response


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "Iris ML API", "version": "3.0.0"}


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "model_loaded": state["model"] is not None,
        "drift_detector_ready": state["drift_detector"] is not None,
    }


@app.get("/metrics")
def metrics() -> Response:
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures) -> PredictionResponse:
    model = state["model"]
    if model is None:
        ERROR_COUNTER.labels(error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model is not loaded")

    x = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])

    try:
        class_id = int(model.predict(x)[0])
        proba = float(model.predict_proba(x)[0, class_id])
    except Exception as exc:
        ERROR_COUNTER.labels(error_type="inference_error").inc()
        logger.exception("inference_failed", extra={"event": "inference_error"})
        raise HTTPException(status_code=500, detail="Inference error") from exc

    class_name = CLASS_NAMES[class_id]
    PREDICTION_COUNTER.labels(class_name=class_name, status="success").inc()
    PREDICTION_CONFIDENCE.observe(proba)

    logger.info(
        "prediction",
        extra={
            "event": "prediction",
            "class_id": class_id,
            "class_name": class_name,
            "probability": round(proba, 4),
            "features": features.model_dump(),
        },
    )

    return PredictionResponse(
        class_id=class_id,
        class_name=class_name,
        probability=round(proba, 4),
    )


@app.post("/check-drift", response_model=DriftResponse)
def check_drift(payload: DriftRequest) -> DriftResponse:
    detector = state["drift_detector"]
    if detector is None:
        ERROR_COUNTER.labels(error_type="drift_detector_not_ready").inc()
        raise HTTPException(status_code=503, detail="Drift detector is not ready")

    DRIFT_CHECKS.inc()
    current = np.array(payload.samples)
    result = detector.detect(current, alpha=payload.alpha)

    for feat, info in result["per_feature"].items():
        if info["drift_detected"]:
            DRIFT_DETECTED.labels(feature=feat).inc()

    logger.info(
        "drift_check",
        extra={
            "event": "drift_check",
            "n_samples": len(payload.samples),
            "alpha": payload.alpha,
            "drift_detected": result["drift_detected"],
            "drifted_features": result["drifted_features"],
        },
    )

    return DriftResponse(**result)
