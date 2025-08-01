import pytest
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock, mock_open

# Add the classifier directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'classifier'))

# Set environment variables to avoid Prefect server connection
os.environ['PREFECT_API_URL'] = 'http://localhost:4200/api'
os.environ['PREFECT_DISABLE_CLIENT'] = 'true'


class TestDataUtils:
    """Test cases for data utility functions."""
    
    def test_load_data(self):
        """Test data loading function."""
        # Create mock data
        mock_X = np.random.rand(100, 130, 13)
        mock_y = np.random.randint(0, 10, 100)
        mock_data = {"mfcc": mock_X, "labels": mock_y}
        
        with patch('numpy.load') as mock_load:
            mock_load.return_value = mock_data
            
            from classifier.utils.data import load_data
            
            X, y = load_data("/path/to/data.npz")
            
            # Verify shape - should add newaxis for channels
            assert X.shape == (100, 130, 13, 1)
            assert y.shape == (100,)
            mock_load.assert_called_once_with("/path/to/data.npz")
    
    @patch('classifier.utils.data.load_data')
    def test_prepare_dataset_splits(self, mock_load):
        """Test dataset preparation and splitting."""
        # Create mock data
        mock_X = np.random.rand(100, 130, 13, 1)
        mock_y = np.random.randint(0, 10, 100)
        mock_load.return_value = (mock_X, mock_y)
        
        from classifier.utils.data import prepare_dataset
        
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset("/path/to/data.npz")
        
        # Verify splits
        total_samples = len(X_train) + len(X_val) + len(X_test)
        assert total_samples == 100
        
        # Test split should be 20% (20 samples)
        assert len(X_test) == 20
        assert len(y_test) == 20
        
        # Validation split should be 25% of remaining 80 samples (20 samples)
        assert len(X_val) == 20
        assert len(y_val) == 20
        
        # Training split should be the remainder (60 samples)
        assert len(X_train) == 60
        assert len(y_train) == 60
        
        # Verify shapes are preserved
        assert X_train.shape[1:] == (130, 13, 1)
        assert X_val.shape[1:] == (130, 13, 1)
        assert X_test.shape[1:] == (130, 13, 1)
    
    @patch('classifier.utils.data.load_data')
    def test_prepare_dataset_custom_splits(self, mock_load):
        """Test dataset preparation with custom split sizes."""
        mock_X = np.random.rand(100, 130, 13, 1)
        mock_y = np.random.randint(0, 10, 100)
        mock_load.return_value = (mock_X, mock_y)
        
        from classifier.utils.data import prepare_dataset
        
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset(
            "/path/to/data.npz", test_size=0.3, val_size=0.2
        )
        
        # Test split: 30%
        assert len(X_test) == 30
        
        # Validation split: 20% of remaining 70 samples = 14 samples
        assert len(X_val) == 14
        
        # Training split: remainder = 56 samples
        assert len(X_train) == 56
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_history(self, mock_subplots, mock_savefig):
        """Test training history plotting."""
        # Setup mock matplotlib
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        
        # Create mock history
        mock_history = Mock()
        mock_history.history = {
            "accuracy": [0.7, 0.8, 0.85, 0.9],
            "val_accuracy": [0.65, 0.75, 0.82, 0.88],
            "loss": [0.8, 0.6, 0.4, 0.2],
            "val_loss": [0.9, 0.7, 0.5, 0.3]
        }
        
        from classifier.utils.data import plot_history
        
        result_fig = plot_history(mock_history, "test_plot.png")
        
        # Verify plot was created
        mock_subplots.assert_called_once_with(2, 1, figsize=(6, 8))
        
        # Verify accuracy plots
        mock_ax1.plot.assert_any_call([0.7, 0.8, 0.85, 0.9], label="Train Accuracy")
        mock_ax1.plot.assert_any_call([0.65, 0.75, 0.82, 0.88], label="Validation Accuracy")
        
        # Verify loss plots
        mock_ax2.plot.assert_any_call([0.8, 0.6, 0.4, 0.2], label="Train Loss")
        mock_ax2.plot.assert_any_call([0.9, 0.7, 0.5, 0.3], label="Validation Loss")
        
        # Verify file saving
        mock_savefig.assert_called_once_with("test_plot.png")
        
        assert result_fig == mock_fig


class TestModel:
    """Test cases for the model building function."""
    
    @patch('tensorflow.keras.Sequential')
    def test_build_model_structure(self, mock_sequential):
        """Test model building with correct structure."""
        mock_model = Mock()
        mock_sequential.return_value = mock_model
        
        from classifier.models.model import build_model
        
        input_shape = (130, 13, 1)
        l2_reg = 0.001
        learning_rate = 0.001
        
        result_model = build_model(input_shape, l2_reg, learning_rate)
        
        # Verify Sequential was called
        mock_sequential.assert_called_once()
        
        # Verify compile was called
        mock_model.compile.assert_called_once()
        
        # Verify compile parameters
        compile_call = mock_model.compile.call_args
        assert compile_call.kwargs['loss'] == "sparse_categorical_crossentropy"
        assert compile_call.kwargs['metrics'] == ["accuracy"]
        
        assert result_model == mock_model
    
    def test_build_model_basic_properties(self):
        """Test model building with basic property checks."""
        from classifier.models.model import build_model
        
        input_shape = (130, 13, 1)
        l2_reg = 0.001
        learning_rate = 0.001
        
        model = build_model(input_shape, l2_reg, learning_rate)
        
        # Verify model structure
        assert len(model.layers) > 0
        assert model.input_shape == (None, 130, 13, 1)
        assert model.output_shape == (None, 10)  # 10 classes
        
        # Verify model loss function (model.loss is a string in newer versions)
        assert model.loss == "sparse_categorical_crossentropy"


class TestTraining:
    """Test cases for the training module."""
    
    @patch('builtins.open', mock_open(read_data="""
experiment_name: "music_genre_classification"
registered_model_name: "music_genre_tf_model"
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  l2_reg: 0.001
  patience: 10
"""))
    @patch('classifier.pipeline.train.mlflow')
    @patch('classifier.pipeline.train.prepare_dataset')
    @patch('classifier.pipeline.train.build_model')
    @patch('classifier.pipeline.train.plot_history')
    def test_train_and_log_model(self, mock_plot, mock_build_model, mock_prepare, mock_mlflow):
        """Test the complete training and logging process."""
        # Setup mock data
        mock_X_train = np.random.rand(60, 130, 13, 1)
        mock_y_train = np.random.randint(0, 10, 60)
        mock_X_val = np.random.rand(20, 130, 13, 1)
        mock_y_val = np.random.randint(0, 10, 20)
        mock_X_test = np.random.rand(20, 130, 13, 1)
        mock_y_test = np.random.randint(0, 10, 20)
        
        mock_prepare.return_value = (mock_X_train, mock_y_train, mock_X_val, mock_y_val, mock_X_test, mock_y_test)
        
        # Setup mock model
        mock_model = Mock()
        mock_history = Mock()
        mock_history.history = {
            "val_accuracy": [0.7, 0.8, 0.85, 0.9]
        }
        mock_model.fit.return_value = mock_history
        mock_model.evaluate.return_value = (0.3, 0.88)  # loss, accuracy
        mock_build_model.return_value = mock_model
        
        # Setup mock plot
        mock_fig = Mock()
        mock_plot.return_value = mock_fig
        
        # Setup MLflow mocks
        mock_run_context = Mock()
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run_context
        mock_mlflow.start_run.return_value.__exit__.return_value = None
        
        from classifier.pipeline.train import train_and_log_model
        
        # Execute training
        train_and_log_model("/path/to/data.npz")
        
        # Verify MLflow setup
        mock_mlflow.set_tracking_uri.assert_called_once_with("http://mlflow:5000")
        mock_mlflow.set_experiment.assert_called_once_with("music_genre_classification")
        
        # Verify model training
        mock_model.fit.assert_called_once()
        fit_call = mock_model.fit.call_args
        assert fit_call.kwargs['epochs'] == 100
        assert fit_call.kwargs['batch_size'] == 32
        
        # Verify model evaluation
        mock_model.evaluate.assert_called_once_with(mock_X_test, mock_y_test, verbose=0)
        
        # Verify MLflow logging
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_metrics.assert_called_once()
        mock_mlflow.log_figure.assert_called_once_with(mock_fig, "training_plot.png")
        mock_mlflow.keras.log_model.assert_called_once()
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_train_and_log_model_missing_config(self, mock_open):
        """Test training fails gracefully when config file is missing."""
        from classifier.pipeline.train import train_and_log_model
        
        with pytest.raises(FileNotFoundError):
            train_and_log_model("/path/to/data.npz")
    
    @patch('yaml.safe_load')
    @patch('builtins.open', mock_open())
    def test_train_and_log_model_invalid_config(self, mock_yaml):
        """Test training handles invalid config gracefully."""
        mock_yaml.side_effect = Exception("Invalid YAML")
        
        from classifier.pipeline.train import train_and_log_model
        
        with pytest.raises(Exception):
            train_and_log_model("/path/to/data.npz")


class TestBasicFunctionality:
    """Test basic functionality without Prefect dependencies."""
    
    @patch('numpy.load')
    @patch('prefect.task', lambda func: func)  # Mock Prefect decorator to do nothing
    def test_feature_extraction_logic(self, mock_load):
        """Test the core feature extraction logic."""
        # Create mock data
        mock_X = np.random.rand(50, 130, 13)
        mock_y = np.random.randint(0, 10, 50)
        mock_data = {"mfcc": mock_X, "labels": mock_y}
        mock_load.return_value = mock_data
        
        from classifier.pipeline.features import extract_features
        
        X, y = extract_features("/path/to/data.npz")
        
        assert X.shape == (50, 130, 13)
        assert y.shape == (50,)
        np.testing.assert_array_equal(X, mock_X)
        np.testing.assert_array_equal(y, mock_y)
        mock_load.assert_called_once_with("/path/to/data.npz")
    
    @patch('os.path.exists')
    @patch('prefect.task', lambda func: func)  # Mock Prefect decorator
    def test_data_validation_logic(self, mock_exists):
        """Test data validation logic."""
        mock_exists.return_value = True
        
        from classifier.pipeline.features import download_and_validate_data
        
        result = download_and_validate_data()
        
        assert result.endswith("data_10.npz")
        assert os.path.isabs(result)
        mock_exists.assert_called_once()
    
    @patch('os.path.exists')
    @patch('prefect.task', lambda func: func)
    def test_data_validation_not_found(self, mock_exists):
        """Test data validation when file not found."""
        mock_exists.return_value = False
        
        from classifier.pipeline.features import download_and_validate_data
        
        with pytest.raises(FileNotFoundError):
            download_and_validate_data()
    
    def test_integration_realistic_shapes(self):
        """Test integration with realistic data shapes."""
        from classifier.models.model import build_model
        
        # Use realistic MFCC dimensions
        input_shape = (130, 13, 1)  # time_steps, mfcc_features, channels
        
        model = build_model(input_shape, l2_reg=0.001, learning_rate=0.001)
        
        # Test with small batch of realistic data
        batch_X = np.random.rand(5, 130, 13, 1)
        batch_y = np.random.randint(0, 10, 5)
        
        # Should not raise errors
        predictions = model.predict(batch_X, verbose=0)
        assert predictions.shape == (5, 10)  # 5 samples, 10 classes
        
        # Test training step
        loss = model.evaluate(batch_X, batch_y, verbose=0)
        assert isinstance(loss, list)  # [loss_value, accuracy_value]
        assert len(loss) == 2


class TestPipelineIntegration:
    """Test integration between pipeline components."""
    
    @patch('numpy.load')
    @patch('os.path.exists')
    @patch('prefect.task', lambda func: func)
    def test_features_to_data_pipeline(self, mock_exists, mock_load):
        """Test integration from features to data preparation."""
        mock_exists.return_value = True
        
        # Create consistent mock data
        mock_X = np.random.rand(100, 130, 13)
        mock_y = np.random.randint(0, 10, 100)
        mock_data = {"mfcc": mock_X, "labels": mock_y}
        mock_load.return_value = mock_data
        
        from classifier.pipeline.features import download_and_validate_data, extract_features
        from classifier.utils.data import prepare_dataset
        
        # Test the pipeline
        data_path = download_and_validate_data()
        X, y = extract_features(data_path)
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset(data_path)
        
        # Verify data flows correctly
        assert X.shape == (100, 130, 13)
        assert y.shape == (100,)
        assert X_train.shape[1:] == (130, 13, 1)  # Channel dimension added
        assert len(X_train) + len(X_val) + len(X_test) == 100
    
    def test_model_data_compatibility(self):
        """Test that model is compatible with prepared data."""
        from classifier.models.model import build_model
        from classifier.utils.data import prepare_dataset
        
        # Create mock data preparation
        with patch('classifier.utils.data.load_data') as mock_load:
            mock_X = np.random.rand(50, 130, 13, 1)
            mock_y = np.random.randint(0, 10, 50)
            mock_load.return_value = (mock_X, mock_y)
            
            X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset("/path/to/data.npz")
            
            # Build model with compatible input shape
            model = build_model(X_train.shape[1:], l2_reg=0.001, learning_rate=0.001)
            
            # Verify model can process the data
            predictions = model.predict(X_train[:5], verbose=0)
            assert predictions.shape == (5, 10)
            
            # Verify model can evaluate
            loss = model.evaluate(X_train[:5], y_train[:5], verbose=0)
            assert len(loss) == 2  # loss and accuracy


if __name__ == "__main__":
    # Quick test runner
    import subprocess
    
    result = subprocess.run([
        "python", "-m", "pytest", __file__, "-v"
    ])
    
    if result.returncode == 0:
        print("✅ All classifier tests passed!")
    else:
        print("❌ Some classifier tests failed")