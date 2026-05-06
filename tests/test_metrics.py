def test_metrics_endpoint_available(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "ml_predictions_total" in response.text
    assert "ml_prediction_latency_seconds" in response.text
    assert "ml_model_loaded" in response.text


def test_predict_increments_counter(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    before = client.get("/metrics").text
    client.post("/predict", json=payload)
    client.post("/predict", json=payload)
    after = client.get("/metrics").text
    assert 'class_name="setosa",status="success"' in after
    assert before != after


def test_check_drift_increments_counter(client):
    healthy = {
        "samples": [
            [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
            [5.4, 3.9, 1.7, 0.4], [5.0, 3.6, 1.4, 0.2], [5.5, 2.5, 4.0, 1.3],
            [6.1, 2.9, 4.7, 1.4], [6.0, 3.0, 4.8, 1.8], [6.3, 2.5, 5.0, 1.9],
            [6.5, 3.0, 5.2, 2.0],
        ],
        "alpha": 0.05,
    }
    response = client.post("/check-drift", json=healthy)
    assert response.status_code == 200
    metrics_text = client.get("/metrics").text
    assert "ml_drift_checks_total" in metrics_text
