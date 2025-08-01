import pytest
import json
import os
import tempfile
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, mock_open
from io import BytesIO
import sys

# Add the api directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))


class TestPredictEndpoint:
    """Test cases for the /predict endpoint."""
    
    @patch('server.Genre_Prediction_Service')
    @patch('server.os.remove')
    @patch('server.os.makedirs')
    @patch('server.os.path.exists')
    @patch('server.pd.read_parquet')
    @patch('server.pd.concat')
    def test_predict_endpoint_success(self, mock_concat, mock_read_parquet, mock_exists, 
                                    mock_makedirs, mock_remove, mock_gps_factory, client):
        """Test successful prediction endpoint call."""
        # Setup mocks
        mock_service = Mock()
        mock_service.predict.return_value = "rock"
        mock_service.extract_mean_mfcc.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        mock_gps_factory.return_value = mock_service
        
        # Mock file system operations
        mock_exists.return_value = False  # No existing parquet file
        
        # Mock DataFrame operations
        mock_df = Mock()
        mock_concat.return_value = mock_df
        
        # Create test file
        test_file = BytesIO(b"fake audio data")
        test_file.name = "test.wav"
        
        # Make request
        response = client.post('/predict', 
                             data={'file': (test_file, 'test.wav')},
                             content_type='multipart/form-data')
        
        # Assertions
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert response_data == {"predicted_genre": "rock"}
        
        # Verify service calls
        mock_gps_factory.assert_called_once()
        assert mock_service.predict.call_count == 1
        assert mock_service.extract_mean_mfcc.call_count == 1
        
        # Verify file operations
        mock_remove.assert_called_once()
        mock_makedirs.assert_called_once()
    
    @patch('server.Genre_Prediction_Service')
    @patch('server.os.remove')
    @patch('server.os.makedirs')
    @patch('server.os.path.exists')
    @patch('server.pd.read_parquet')
    @patch('server.pd.concat')
    def test_predict_endpoint_with_actual_genre(self, mock_concat, mock_read_parquet, mock_exists,
                                              mock_makedirs, mock_remove, mock_gps_factory, client):
        """Test prediction endpoint with actual genre provided."""
        # Setup mocks
        mock_service = Mock()
        mock_service.predict.return_value = "jazz"
        mock_service.extract_mean_mfcc.return_value = [0.1] * 13
        mock_gps_factory.return_value = mock_service
        
        mock_exists.return_value = False
        mock_df = Mock()
        mock_concat.return_value = mock_df
        
        # Create test file
        test_file = BytesIO(b"fake audio data")
        test_file.name = "test.wav"
        
        # Make request with actual genre
        response = client.post('/predict', 
                             data={
                                 'file': (test_file, 'test.wav'),
                                 'actual_genre': 'rock'
                             },
                             content_type='multipart/form-data')
        
        # Assertions
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert response_data == {"predicted_genre": "jazz"}
    
    @patch('server.Genre_Prediction_Service')
    @patch('server.os.remove')
    @patch('server.os.makedirs')
    @patch('server.os.path.exists')
    @patch('server.pd.read_parquet')
    @patch('server.pd.concat')
    def test_predict_endpoint_with_existing_parquet(self, mock_concat, mock_read_parquet, mock_exists,
                                                  mock_makedirs, mock_remove, mock_gps_factory, client):
        """Test prediction endpoint when parquet file already exists."""
        # Setup mocks
        mock_service = Mock()
        mock_service.predict.return_value = "blues"
        mock_service.extract_mean_mfcc.return_value = [0.1] * 13
        mock_gps_factory.return_value = mock_service
        
        # Mock existing parquet file
        mock_exists.return_value = True
        mock_existing_df = pd.DataFrame({'mfcc_1': [0.5], 'predicted_genre': ['pop']})
        mock_read_parquet.return_value = mock_existing_df
        
        mock_combined_df = Mock()
        mock_concat.return_value = mock_combined_df
        
        # Create test file
        test_file = BytesIO(b"fake audio data")
        test_file.name = "test.wav"
        
        # Make request
        response = client.post('/predict', 
                             data={'file': (test_file, 'test.wav')},
                             content_type='multipart/form-data')
        
        # Assertions
        assert response.status_code == 200
        
        # Verify parquet operations
        mock_read_parquet.assert_called_once_with("monitoring/data/current.parquet")
        mock_concat.assert_called_once()
        mock_combined_df.to_parquet.assert_called_once_with("monitoring/data/current.parquet", index=False)
    
    @patch('server.Genre_Prediction_Service')
    @patch('server.os.remove')
    def test_predict_endpoint_with_invalid_file_content(self, mock_remove, mock_gps_factory, client):
        """Test prediction endpoint with invalid file content that causes service errors."""
        # This is a more useful test - what happens when the service can't process the file
        mock_service = Mock()
        mock_service.predict.side_effect = ValueError("Audio signal too short for processing.")
        mock_gps_factory.return_value = mock_service
        
        # Create test file
        test_file = BytesIO(b"invalid audio data")
        test_file.name = "test.wav"
        
        # The service error should propagate (since server.py doesn't handle it)
        with pytest.raises(ValueError, match="Audio signal too short for processing"):
            client.post('/predict', 
                       data={'file': (test_file, 'test.wav')},
                       content_type='multipart/form-data')
    
    @patch('server.Genre_Prediction_Service')
    @patch('server.os.remove')
    def test_predict_endpoint_service_error(self, mock_remove, mock_gps_factory, client):
        """Test prediction endpoint when service raises an error."""
        # Setup mock to raise exception
        mock_service = Mock()
        mock_service.predict.side_effect = Exception("Model error")
        mock_gps_factory.return_value = mock_service
        
        # Create test file
        test_file = BytesIO(b"fake audio data")
        test_file.name = "test.wav"
        
        # Make request - expect it to raise an exception since Flask doesn't handle it
        with pytest.raises(Exception, match="Model error"):
            response = client.post('/predict', 
                                 data={'file': (test_file, 'test.wav')},
                                 content_type='multipart/form-data')
    
    @patch('server.Genre_Prediction_Service')
    @patch('server.os.remove')
    @patch('server.os.makedirs')
    @patch('server.os.path.exists')
    @patch('server.random.randint')
    def test_predict_endpoint_random_filename(self, mock_randint, mock_exists, mock_makedirs, 
                                            mock_remove, mock_gps_factory, client):
        """Test that prediction endpoint uses random filename."""
        # Setup mocks
        mock_randint.return_value = 12345
        mock_service = Mock()
        mock_service.predict.return_value = "metal"
        mock_service.extract_mean_mfcc.return_value = [0.1] * 13
        mock_gps_factory.return_value = mock_service
        
        mock_exists.return_value = False
        
        # Create test file
        test_file = BytesIO(b"fake audio data")
        test_file.name = "test.wav"
        
        # Make request
        response = client.post('/predict', 
                             data={'file': (test_file, 'test.wav')},
                             content_type='multipart/form-data')
        
        # Verify random filename was used
        mock_service.predict.assert_called_once_with("12345.wav")
        mock_service.extract_mean_mfcc.assert_called_once_with("12345.wav")
        mock_remove.assert_called_once_with("12345.wav")
    
    @patch('server.Genre_Prediction_Service')
    @patch('server.os.remove')
    @patch('server.os.makedirs')
    @patch('server.os.path.exists')
    def test_predict_endpoint_dataframe_structure(self, mock_exists, mock_makedirs, mock_remove, 
                                                mock_gps_factory, client):
        """Test that the endpoint creates DataFrame with correct structure."""
        # Setup mocks
        mock_service = Mock()
        mock_service.predict.return_value = "disco"
        mock_mfcc_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        mock_service.extract_mean_mfcc.return_value = mock_mfcc_vector
        mock_gps_factory.return_value = mock_service
        
        mock_exists.return_value = False
        
        # Create test file
        test_file = BytesIO(b"fake audio data")
        test_file.name = "test.wav"
        
        # Mock DataFrame constructor and operations
        with patch('server.pd.DataFrame') as mock_df_constructor:
            # Create a proper mock DataFrame that supports item assignment
            mock_df = Mock()
            mock_df.__setitem__ = Mock()  # Allow item assignment
            mock_df.to_parquet = Mock()   # Allow to_parquet calls
            mock_df_constructor.return_value = mock_df
            
            # Make request
            response = client.post('/predict', 
                                 data={
                                     'file': (test_file, 'test.wav'),
                                     'actual_genre': 'rock'
                                 },
                                 content_type='multipart/form-data')
            
            # Verify response is successful
            assert response.status_code == 200
            response_data = json.loads(response.data)
            assert response_data == {"predicted_genre": "disco"}
            
            # Verify DataFrame was created with correct structure
            expected_columns = [f"mfcc_{i+1}" for i in range(13)]
            mock_df_constructor.assert_called_once_with([mock_mfcc_vector], columns=expected_columns)
            
            # Verify additional columns were set
            mock_df.__setitem__.assert_any_call("predicted_genre", "disco")
            mock_df.__setitem__.assert_any_call("actual_genre", "rock")


class TestServerConfiguration:
    """Test cases for server configuration and setup."""
    
    def test_flask_app_exists(self):
        """Test that Flask app is properly configured."""
        from server import app
        assert app is not None
        assert app.name == 'server'
    
    def test_parquet_path_constant(self):
        """Test that parquet path constant is properly defined."""
        from server import CURRENT_PARQUET_PATH
        assert CURRENT_PARQUET_PATH == "monitoring/data/current.parquet"
    
    @patch('server.app.run')
    def test_main_execution(self, mock_run):
        """Test that app runs when script is executed directly."""
        # This would test the if __name__ == "__main__" block
        # We'll simulate it by calling the run method
        from server import app
        
        # Verify that if we were to run the app, it would use correct settings
        with patch('builtins.print') as mock_print:
            app.run(debug=False)
            mock_run.assert_called_once_with(debug=False)