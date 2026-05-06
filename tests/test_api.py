def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "3.0.0"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["drift_detector_ready"] is True


def test_predict_setosa(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["class_name"] == "setosa"
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_virginica(client):
    payload = {
        "sepal_length": 6.7,
        "sepal_width": 3.0,
        "petal_length": 5.2,
        "petal_width": 2.3,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["class_name"] == "virginica"


def test_predict_invalid_input(client):
    response = client.post("/predict", json={"sepal_length": "not-a-number"})
    assert response.status_code == 422


def test_predict_out_of_range(client):
    payload = {
        "sepal_length": -1.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
