import pytest
from fastapi.testclient import TestClient

from app.main import MODEL_PATH, REFERENCE_PATH, app
from ml.train import train_and_save

if not MODEL_PATH.exists() or not REFERENCE_PATH.exists():
    train_and_save(MODEL_PATH, REFERENCE_PATH)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
