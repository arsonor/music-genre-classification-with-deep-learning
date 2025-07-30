import os
import pytest
from unittest.mock import MagicMock

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """
    Sets up common test environment variables for MLflow & Prefect
    without touching production services.
    """
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://mock-mlflow:5000")
    os.environ.setdefault("PREFECT_API_URL", "http://mock-prefect:4200/api")
    os.environ.setdefault("API_ENV", "test")
    yield

@pytest.fixture
def mock_mlflow(mocker):
    """
    Provides a mock MLflow client so tests don't hit a real MLflow server.
    """
    mock_client = MagicMock()
    mocker.patch("mlflow.tracking.MlflowClient", return_value=mock_client)
    return mock_client

@pytest.fixture
def mock_prefect(mocker):
    """
    Mocks Prefect tasks so the flow doesn't actually run.
    """
    mocker.patch("prefect.task_runners.ConcurrentTaskRunner")
    mocker.patch("prefect.flows.Flow")
    return True

@pytest.fixture
def mock_requests(mocker):
    """
    Mocks 'requests' to avoid real HTTP calls in inference tests.
    """
    return mocker.patch("requests.post")

@pytest.fixture
def sample_audio_file(tmp_path):
    """
    Creates a tiny fake WAV file for API upload tests.
    """
    fake_file = tmp_path / "test.wav"
    fake_file.write_bytes(b"FAKEAUDIOCONTENT")
    return open(fake_file, "rb")

