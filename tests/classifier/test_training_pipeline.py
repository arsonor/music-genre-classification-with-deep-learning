from unittest.mock import patch
from classifier.prefect_flow import music_genre_pipeline

@patch("mlflow.start_run")
def test_pipeline_runs_without_errors(mock_mlflow):
    # Run the flow locally to ensure no import errors
    result = music_genre_pipeline()
    assert result is not None