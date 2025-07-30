import json
import pytest
from unittest.mock import patch

@pytest.fixture
def client():
    # Import inside fixture to ensure env vars from conftest are applied first
    from api.server import app
    app.testing = True
    return app.test_client()

def test_predict_endpoint_success(client, mock_mlflow):
    """
    Test /predict returns correct format and status code.
    """
    mock_prediction = ["rock"]
    with patch("api.genre_prediction_service.Genre_Prediction_Service") as mock_service:
        mock_service.return_value.predict.return_value = mock_prediction

        audio_file = (pytest.lazy_fixture("sample_audio_file"), "test.wav")
        data = {"file": audio_file}
        response = client.post("/predict", data=data, content_type="multipart/form-data")

    assert response.status_code == 200
    data = json.loads(response.data)
    assert "prediction" in data
    assert data["prediction"] == mock_prediction

def test_predict_endpoint_missing_file(client):
    """
    Test /predict without sending a file should return 400.
    """
    response = client.post("/predict")
    assert response.status_code == 400

