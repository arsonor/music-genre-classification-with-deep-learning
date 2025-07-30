import pytest
from unittest.mock import patch

def test_genre_prediction_service_loads_model(mock_mlflow):
    """
    Ensure the Genre_Prediction_Service tries to load the MLflow model on init.
    """
    with patch("mlflow.keras.load_model") as mock_load:
        from api.genre_prediction_service import Genre_Prediction_Service
        Genre_Prediction_Service()
        mock_load.assert_called_once()

def test_genre_prediction_service_predict(mock_mlflow):
    """
    Ensure predict() returns expected genre string.
    """
    mock_model = type("MockModel", (), {
        "predict": lambda self, x: [[0.1, 0.9]]
    })()

    with patch("mlflow.keras.load_model", return_value=mock_model):
        from api.genre_prediction_service import Genre_Prediction_Service
        service = Genre_Prediction_Service()
        prediction = service.predict("fake_audio_path")

    assert isinstance(prediction, list)
    assert len(prediction) == 1
