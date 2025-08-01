import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

# Import will be done in individual test files to avoid import issues


@pytest.fixture
def mock_audio_data():
    """Generate mock audio data for testing."""
    # Create 3 seconds of mock audio data at 22050 Hz sample rate
    samples = 22050 * 3
    return np.random.randn(samples).astype(np.float32)


@pytest.fixture
def mock_mfcc_data():
    """Generate mock MFCC data for testing."""
    # Standard MFCC shape: (13, time_frames)
    return np.random.randn(13, 130).astype(np.float32)


@pytest.fixture
def mock_mfcc_transposed():
    """Generate mock transposed MFCC data for testing."""
    # Transposed MFCC shape: (time_frames, 13)
    return np.random.randn(130, 13).astype(np.float32)


@pytest.fixture
def mock_mfcc_mean():
    """Generate mock mean MFCC vector for testing."""
    return np.random.randn(13).astype(np.float32)


@pytest.fixture
def mock_model_predictions():
    """Generate mock model predictions for testing."""
    # 10 genres, so 10 probability scores
    predictions = np.random.rand(1, 10).astype(np.float32)
    # Make one prediction clearly dominant
    predictions[0, 3] = 0.9  # disco (index 3)
    return predictions


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict.return_value = np.array([[0.1, 0.05, 0.15, 0.9, 0.02, 0.08, 0.12, 0.03, 0.06, 0.04]])
    return model


@pytest.fixture
def mock_genre_service_instance(mock_model):
    """Create a mock genre service instance."""
    # Import here to avoid circular imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))
    from genre_prediction_service import _Genre_Prediction_Service
    
    service = _Genre_Prediction_Service()
    service.model = mock_model
    return service


@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        # Write some dummy data to make it a valid file
        f.write(b'dummy audio data')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_parquet_file():
    """Create a temporary parquet file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        temp_path = f.name
    
    # Remove the file so tests can create it
    os.remove(temp_path)
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    data = {
        f'mfcc_{i+1}': [np.random.randn()] for i in range(13)
    }
    data.update({
        'predicted_genre': ['rock'],
        'actual_genre': ['rock']
    })
    return pd.DataFrame(data)


@pytest.fixture
def flask_app():
    """Create a Flask app instance for testing."""
    # Import here to avoid circular imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))
    from server import app
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(flask_app):
    """Create a test client for the Flask app."""
    return flask_app.test_client()


@pytest.fixture
def mock_librosa_load():
    """Mock librosa.load function."""
    def _mock_load(file_path, sr=None):
        # Return mock audio data and sample rate
        samples = 22050 * 3  # 3 seconds of audio
        return np.random.randn(samples).astype(np.float32), 22050
    return _mock_load


@pytest.fixture
def mock_librosa_mfcc():
    """Mock librosa.feature.mfcc function."""
    def _mock_mfcc(y, sr, n_mfcc=13, n_fft=2048, hop_length=512):
        # Return mock MFCC data with shape (n_mfcc, time_frames)
        time_frames = len(y) // hop_length + 1
        return np.random.randn(n_mfcc, time_frames).astype(np.float32)
    return _mock_mfcc


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton instance before each test."""
    # Import here to avoid issues
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))
    try:
        from genre_prediction_service import _Genre_Prediction_Service
        _Genre_Prediction_Service._instance = None
        _Genre_Prediction_Service.model = None
        yield
        # Cleanup after test
        _Genre_Prediction_Service._instance = None
        _Genre_Prediction_Service.model = None
    except ImportError:
        # If import fails, just yield
        yield