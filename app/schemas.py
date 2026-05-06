from typing import Dict, List

from pydantic import BaseModel, Field, conlist


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10, description="cm")
    sepal_width: float = Field(..., ge=0, le=10, description="cm")
    petal_length: float = Field(..., ge=0, le=10, description="cm")
    petal_width: float = Field(..., ge=0, le=10, description="cm")


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    probability: float


class DriftRequest(BaseModel):
    samples: conlist(conlist(float, min_length=4, max_length=4), min_length=10)
    alpha: float = Field(default=0.05, ge=0.001, le=0.5)


class FeatureDriftInfo(BaseModel):
    statistic: float
    p_value: float
    drift_detected: bool


class DriftResponse(BaseModel):
    drift_detected: bool
    n_drifted_features: int
    drifted_features: List[str]
    per_feature: Dict[str, FeatureDriftInfo]
    n_samples: int
    alpha: float
