import pytest
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add the api directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

from genre_prediction_service import _Genre_Prediction_Service, Genre_Prediction_Service


class TestGenrePredictionServiceClass:
    """Test cases for the _Genre_Prediction_Service class methods."""
    
    def test_mapping_contains_all_genres(self):
        """Test that the mapping contains all expected genres."""
        expected_genres = [
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock"
        ]
        service = _Genre_Prediction_Service()
        assert service._mapping == expected_genres
        assert len(service._mapping) == 10
    
    @patch('genre_prediction_service.librosa.load')
    @patch('genre_prediction_service.librosa.feature.mfcc')
    def test_preprocess_valid_audio(self, mock_mfcc, mock_load):
        """Test preprocessing with valid audio data."""
        # Create predictable test data
        test_audio = np.random.rand(66150).astype(np.float32)  # 3 seconds
        test_mfcc = np.random.rand(13, 130).astype(np.float32)  # 13 coeffs, 130 frames
        
        # Setup mocks
        mock_load.return_value = (test_audio, 22050)
        mock_mfcc.return_value = test_mfcc
        
        service = _Genre_Prediction_Service()
        result = service.preprocess("dummy_path.wav")
        
        # Verify librosa functions were called - but avoid direct array comparison in assertions
        mock_load.assert_called_once_with("dummy_path.wav")
        
        # Check that mfcc was called once, and verify the parameters separately
        mock_mfcc.assert_called_once()
        call_args = mock_mfcc.call_args
        assert call_args.kwargs['sr'] == 22050
        assert call_args.kwargs['n_mfcc'] == 13
        assert call_args.kwargs['n_fft'] == 2048
        assert call_args.kwargs['hop_length'] == 512
        # Verify the audio array shape and type
        assert call_args.kwargs['y'].shape == test_audio.shape
        assert call_args.kwargs['y'].dtype == test_audio.dtype
        
        # Verify result shape and content
        assert result.shape == (130, 13)  # Transposed
        np.testing.assert_array_equal(result, test_mfcc.T)
    
    @patch('genre_prediction_service.librosa.load')
    def test_preprocess_short_audio_raises_error(self, mock_load):
        """Test that preprocessing raises error for short audio."""
        # Mock short audio (less than 3 seconds)
        short_audio = np.random.rand(1000).astype(np.float32)
        mock_load.return_value = (short_audio, 22050)
        
        service = _Genre_Prediction_Service()
        
        with pytest.raises(ValueError, match="Audio signal too short for processing"):
            service.preprocess("short_audio.wav")
    
    @patch('genre_prediction_service.librosa.load')
    @patch('genre_prediction_service.librosa.feature.mfcc')
    def test_preprocess_custom_parameters(self, mock_mfcc, mock_load):
        """Test preprocessing with custom parameters."""
        test_audio = np.random.rand(66150).astype(np.float32)
        test_mfcc = np.random.rand(20, 100).astype(np.float32)  # Custom dimensions
        
        mock_load.return_value = (test_audio, 22050)
        mock_mfcc.return_value = test_mfcc
        
        service = _Genre_Prediction_Service()
        result = service.preprocess("dummy_path.wav", num_mfcc=20, n_fft=1024, hop_length=256)
        
        # Verify the call was made with correct parameters (avoid array comparison)
        mock_mfcc.assert_called_once()
        call_args = mock_mfcc.call_args
        assert call_args.kwargs['sr'] == 22050
        assert call_args.kwargs['n_mfcc'] == 20
        assert call_args.kwargs['n_fft'] == 1024
        assert call_args.kwargs['hop_length'] == 256
        assert call_args.kwargs['y'].shape == test_audio.shape
        
        assert result.shape == (100, 20)  # Transposed custom dimensions
    
    @patch('genre_prediction_service.librosa.load')
    @patch('genre_prediction_service.librosa.feature.mfcc')
    def test_extract_mean_mfcc(self, mock_mfcc, mock_load):
        """Test MFCC mean extraction."""
        test_audio = np.random.rand(66150).astype(np.float32)
        test_mfcc = np.random.rand(13, 130).astype(np.float32)
        
        mock_load.return_value = (test_audio, 22050)
        mock_mfcc.return_value = test_mfcc
        
        service = _Genre_Prediction_Service()
        result = service.extract_mean_mfcc("dummy_path.wav")
        
        # Verify result is mean along axis 1
        expected_mean = np.mean(test_mfcc, axis=1)
        np.testing.assert_array_almost_equal(result, expected_mean)
        assert result.shape == (13,)
    
    @patch('genre_prediction_service.librosa.load')
    @patch('genre_prediction_service.librosa.feature.mfcc')
    def test_extract_mean_mfcc_custom_params(self, mock_mfcc, mock_load):
        """Test MFCC mean extraction with custom parameters."""
        test_audio = np.random.rand(66150).astype(np.float32)
        test_mfcc = np.random.rand(15, 100).astype(np.float32)
        
        mock_load.return_value = (test_audio, 22050)
        mock_mfcc.return_value = test_mfcc
        
        service = _Genre_Prediction_Service()
        result = service.extract_mean_mfcc("dummy_path.wav", n_mfcc=15, n_fft=1024, hop_length=256)
        
        # Verify the call parameters without array comparison
        mock_mfcc.assert_called_once()
        call_args = mock_mfcc.call_args
        assert call_args.kwargs['sr'] == 22050
        assert call_args.kwargs['n_mfcc'] == 15
        assert call_args.kwargs['n_fft'] == 1024
        assert call_args.kwargs['hop_length'] == 256
        assert call_args.kwargs['y'].shape == test_audio.shape
        
        expected_mean = np.mean(test_mfcc, axis=1)
        np.testing.assert_array_almost_equal(result, expected_mean)
        assert result.shape == (15,)
    
    def test_predict_returns_correct_genre(self):
        """Test that predict returns the correct genre."""
        service = _Genre_Prediction_Service()
        
        # Mock the model and preprocess method
        mock_model = Mock()
        mock_predictions = np.array([[0.1, 0.05, 0.15, 0.9, 0.02, 0.08, 0.12, 0.03, 0.06, 0.04]])
        mock_model.predict.return_value = mock_predictions
        service.model = mock_model
        
        # Mock preprocessed data
        mock_features = np.random.rand(130, 13).astype(np.float32)
        
        with patch.object(service, 'preprocess', return_value=mock_features) as mock_preprocess:
            result = service.predict("dummy_path.wav")
            
            # The mock model returns highest probability for index 3 (disco)
            assert result == "disco"
            
            # Verify preprocess was called
            mock_preprocess.assert_called_once_with("dummy_path.wav")
            
            # Verify model.predict was called with correct shape
            called_args = mock_model.predict.call_args[0][0]
            expected_shape = (1, 130, 13, 1)
            assert called_args.shape == expected_shape
    
    def test_predict_reshapes_input_correctly(self):
        """Test that predict reshapes input correctly for the model."""
        service = _Genre_Prediction_Service()
        
        mock_model = Mock()
        mock_predictions = np.array([[0.1, 0.05, 0.15, 0.02, 0.03, 0.9, 0.12, 0.03, 0.06, 0.04]])
        mock_model.predict.return_value = mock_predictions
        service.model = mock_model
        
        mock_features = np.random.rand(130, 13).astype(np.float32)
        
        with patch.object(service, 'preprocess', return_value=mock_features):
            service.predict("dummy_path.wav")
            
            # Verify the input was reshaped correctly
            called_args = mock_model.predict.call_args[0][0]
            assert len(called_args.shape) == 4
            assert called_args.shape[0] == 1  # batch size
            assert called_args.shape[-1] == 1  # channels
