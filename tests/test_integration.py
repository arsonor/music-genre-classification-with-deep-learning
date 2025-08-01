import pytest
import numpy as np
import os
import tempfile
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
import sys

# Add the api directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))


@pytest.mark.integration
class TestEndToEndIntegration:
    """Integration tests that test the complete flow from API to service."""
    
    @patch('server.Genre_Prediction_Service')
    @patch('librosa.load')
    @patch('librosa.feature.mfcc')
    @patch('server.os.remove')
    @patch('server.os.makedirs')
    @patch('server.os.path.exists')
    def test_complete_prediction_flow(self, mock_exists, mock_makedirs, mock_remove,
                                    mock_mfcc, mock_load, mock_gps_factory, client):
        """Test complete flow from API request to response."""
        # Setup realistic mock data
        mock_audio_data = np.random.randn(66150).astype(np.float32)  # 3 seconds at 22050 Hz
        mock_mfcc_data = np.random.randn(13, 130).astype(np.float32)
        
        mock_load.return_value = (mock_audio_data, 22050)
        mock_mfcc.return_value = mock_mfcc_data
        
        # Create a real service instance but with mocked dependencies
        from genre_prediction_service import _Genre_Prediction_Service
        real_service = _Genre_Prediction_Service()
        
        # Mock the model
        mock_model = Mock()
        mock_predictions = np.array([[0.1, 0.05, 0.15, 0.02, 0.03, 0.9, 0.12, 0.03, 0.06, 0.04]])
        mock_model.predict.return_value = mock_predictions
        real_service.model = mock_model
        
        mock_gps_factory.return_value = real_service
        mock_exists.return_value = False
        
        # Create test file
        test_file = BytesIO(b"fake audio data")
        test_file.name = "test.wav"
        
        # Make request
        response = client.post('/predict', 
                             data={
                                 'file': (test_file, 'test.wav'),
                                 'actual_genre': 'jazz'
                             },
                             content_type='multipart/form-data')
        
        # Verify response
        assert response.status_code == 200
        response_data = response.get_json()
        assert response_data['predicted_genre'] == 'jazz'  # Index 5 in mapping
        
        # Verify the complete chain was called
        mock_load.assert_called()
        mock_mfcc.assert_called()
        mock_model.predict.assert_called_once()
        
        # Verify file cleanup
        mock_remove.assert_called_once()
    
    @patch('server.Genre_Prediction_Service')
    @patch('librosa.load')
    @patch('librosa.feature.mfcc')
    @patch('server.os.remove')
    @patch('server.os.makedirs')
    @patch('server.os.path.exists')
    @patch('server.pd.read_parquet')
    @patch('server.pd.concat')
    def test_monitoring_data_persistence(self, mock_concat, mock_read_parquet, mock_exists,
                                       mock_makedirs, mock_remove, mock_mfcc, mock_load,
                                       mock_gps_factory, client, temp_parquet_file):
        """Test that monitoring data is correctly persisted."""
        # Setup mocks
        mock_audio_data = np.random.randn(66150).astype(np.float32)
        mock_mfcc_data = np.random.randn(13, 130).astype(np.float32)
        
        mock_load.return_value = (mock_audio_data, 22050)
        mock_mfcc.return_value = mock_mfcc_data
        
        # Create service with mocked model
        from genre_prediction_service import _Genre_Prediction_Service
        real_service = _Genre_Prediction_Service()
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.1, 0.05, 0.15, 0.02, 0.03, 0.9, 0.12, 0.03, 0.06, 0.04]])
        real_service.model = mock_model
        mock_gps_factory.return_value = real_service
        
        # Mock existing parquet file
        existing_data = pd.DataFrame({
            'mfcc_1': [0.1], 'mfcc_2': [0.2], 'mfcc_3': [0.3], 'mfcc_4': [0.4],
            'mfcc_5': [0.5], 'mfcc_6': [0.6], 'mfcc_7': [0.7], 'mfcc_8': [0.8],
            'mfcc_9': [0.9], 'mfcc_10': [1.0], 'mfcc_11': [1.1], 'mfcc_12': [1.2], 'mfcc_13': [1.3],
            'predicted_genre': ['rock'], 'actual_genre': ['rock']
        })
        
        mock_exists.return_value = True
        mock_read_parquet.return_value = existing_data
        
        # Setup concat to return combined data
        combined_data = Mock()
        mock_concat.return_value = combined_data
        
        # Create test file
        test_file = BytesIO(b"fake audio data")
        test_file.name = "test.wav"
        
        # Make request
        response = client.post('/predict', 
                             data={
                                 'file': (test_file, 'test.wav'),
                                 'actual_genre': 'blues'
                             },
                             content_type='multipart/form-data')
        
        # Verify response
        assert response.status_code == 200
        
        # Verify monitoring data operations
        mock_read_parquet.assert_called_once_with("monitoring/data/current.parquet")
        mock_concat.assert_called_once()
        combined_data.to_parquet.assert_called_once_with("monitoring/data/current.parquet", index=False)
    
    @patch('server.Genre_Prediction_Service')
    @patch('librosa.load')
    def test_error_handling_short_audio(self, mock_load, mock_gps_factory, client):
        """Test error handling when audio is too short."""
        # Setup short audio that will cause ValueError
        short_audio = np.random.randn(1000).astype(np.float32)  # Less than 3 seconds
        mock_load.return_value = (short_audio, 22050)
        
        # Create real service
        from genre_prediction_service import _Genre_Prediction_Service
        real_service = _Genre_Prediction_Service()
        mock_model = Mock()
        real_service.model = mock_model
        mock_gps_factory.return_value = real_service
        
        # Create test file
        test_file = BytesIO(b"fake audio data")
        test_file.name = "test.wav"
        
        # Make request - should raise ValueError since the signal is too short
        with pytest.raises(ValueError, match="Audio signal too short for processing"):
            response = client.post('/predict', 
                                 data={'file': (test_file, 'test.wav')},
                                 content_type='multipart/form-data')


@pytest.mark.integration
class TestServiceIntegration:
    """Integration tests for service components."""
    
    @patch('librosa.load')
    @patch('librosa.feature.mfcc')
    def test_preprocess_and_predict_integration(self, mock_mfcc, mock_load):
        """Test integration between preprocess and predict methods."""
        # Setup realistic mock data
        mock_audio_data = np.random.randn(66150).astype(np.float32)
        mock_mfcc_data = np.random.randn(13, 130).astype(np.float32)
        
        mock_load.return_value = (mock_audio_data, 22050)
        mock_mfcc.return_value = mock_mfcc_data
        
        # Create service with mock model
        from genre_prediction_service import _Genre_Prediction_Service
        service = _Genre_Prediction_Service()
        
        # Mock model
        mock_model = Mock()
        mock_predictions = np.array([[0.1, 0.05, 0.15, 0.02, 0.03, 0.9, 0.12, 0.03, 0.06, 0.04]])
        mock_model.predict.return_value = mock_predictions
        service.model = mock_model
        
        # Test the integration
        result = service.predict("dummy_path.wav")
        
        # Verify the result
        assert result == "jazz"  # Index 5 in the mapping
        
        # Verify the preprocessing was called correctly
        mock_load.assert_called_once_with("dummy_path.wav")
        mock_mfcc.assert_called_once()
        
        # Verify model input shape
        model_input = mock_model.predict.call_args[0][0]
        assert model_input.shape == (1, 130, 13, 1)  # batch, time, features, channels
    
    @patch('librosa.load')
    @patch('librosa.feature.mfcc')
    def test_extract_mfcc_methods_consistency(self, mock_mfcc, mock_load):
        """Test that both MFCC extraction methods are consistent."""
        # Setup mock data
        mock_audio_data = np.random.randn(66150).astype(np.float32)
        mock_mfcc_data = np.random.randn(13, 130).astype(np.float32)
        
        mock_load.return_value = (mock_audio_data, 22050)
        mock_mfcc.return_value = mock_mfcc_data
        
        # Create service
        from genre_prediction_service import _Genre_Prediction_Service
        service = _Genre_Prediction_Service()
        
        # Test both methods
        mfcc_from_preprocess = service.preprocess("dummy_path.wav")
        
        # Reset mocks for second call
        mock_load.reset_mock()
        mock_mfcc.reset_mock()
        mock_load.return_value = (mock_audio_data, 22050)
        mock_mfcc.return_value = mock_mfcc_data
        
        mfcc_mean = service.extract_mean_mfcc("dummy_path.wav")
        
        # Verify consistency
        expected_mean = np.mean(mfcc_from_preprocess, axis=0)
        np.testing.assert_array_almost_equal(mfcc_mean, expected_mean, decimal=5)


@pytest.mark.integration
class TestFactoryIntegration:
    """Integration tests for the factory function."""
    
    @patch('librosa.load')
    @patch('librosa.feature.mfcc')
    def test_service_class_to_prediction_integration(self, mock_mfcc, mock_load):
        """Test integration from service class creation to actual prediction."""
        # Setup librosa mocks
        mock_audio_data = np.random.randn(66150).astype(np.float32)
        mock_mfcc_data = np.random.randn(13, 130).astype(np.float32)
        mock_load.return_value = (mock_audio_data, 22050)
        mock_mfcc.return_value = mock_mfcc_data
        
        # Create service with mock model
        from genre_prediction_service import _Genre_Prediction_Service
        service = _Genre_Prediction_Service()
        
        # Mock model
        mock_model = Mock()
        mock_predictions = np.array([[0.1, 0.05, 0.15, 0.02, 0.03, 0.9, 0.12, 0.03, 0.06, 0.04]])
        mock_model.predict.return_value = mock_predictions
        service.model = mock_model
        
        # Test the integration
        result = service.predict("dummy_path.wav")
        
        # Verify the result
        assert result == "jazz"  # Index 5 in the mapping
        
        # Verify the preprocessing was called correctly
        mock_load.assert_called_once_with("dummy_path.wav")
        mock_mfcc.assert_called_once()
        
        # Verify model input shape
        model_input = mock_model.predict.call_args[0][0]
        assert model_input.shape == (1, 130, 13, 1)  # batch, time, features, channels
    
    def test_singleton_behavior_integration(self):
        """Test that singleton behavior works correctly in integration context."""
        from genre_prediction_service import _Genre_Prediction_Service
        
        # Create two instances of the service class
        service1 = _Genre_Prediction_Service()
        service2 = _Genre_Prediction_Service()
        
        # They should be different instances (not singleton at class level)
        # The singleton is implemented at the factory function level
        assert service1 is not service2
        
        # But they should have the same structure
        assert hasattr(service1, 'predict')
        assert hasattr(service1, 'preprocess')
        assert hasattr(service1, 'extract_mean_mfcc')
        assert hasattr(service1, '_mapping')
        
        # Verify the mapping is consistent
        assert service1._mapping == service2._mapping
        assert len(service1._mapping) == 10