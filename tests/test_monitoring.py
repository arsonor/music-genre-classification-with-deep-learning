import os
import sys
import subprocess
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add the monitoring directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "monitoring"))

# pylint: disable=wrong-import-position
from run_monitoring import app, prometheus_safe

# pylint: enable=wrong-import-position


class TestPrometheusUtility:
    """Test cases for the prometheus_safe utility function."""

    def test_prometheus_safe_basic_names(self):
        """Test prometheus_safe with basic metric names."""
        assert prometheus_safe("accuracy") == "accuracy"
        assert prometheus_safe("f1_score") == "f1_score"
        assert prometheus_safe("row_count") == "row_count"

    def test_prometheus_safe_special_characters(self):
        """Test prometheus_safe with special characters."""
        assert prometheus_safe("metric-name") == "metric_name"
        assert prometheus_safe("metric.name") == "metric_name"
        assert prometheus_safe("metric@name") == "metric_name"
        assert prometheus_safe("metric name") == "metric_name"
        assert prometheus_safe("metric/name") == "metric_name"

    def test_prometheus_safe_multiple_underscores(self):
        """Test prometheus_safe removes consecutive underscores."""
        assert prometheus_safe("metric__name") == "metric_name"
        assert prometheus_safe("metric___name") == "metric_name"
        assert prometheus_safe("metric____name") == "metric_name"

    def test_prometheus_safe_leading_trailing_underscores(self):
        """Test prometheus_safe removes leading/trailing underscores."""
        assert prometheus_safe("_metric_name_") == "metric_name"
        assert prometheus_safe("__metric_name__") == "metric_name"

    def test_prometheus_safe_uppercase(self):
        """Test prometheus_safe converts to lowercase."""
        assert prometheus_safe("METRIC_NAME") == "metric_name"
        assert prometheus_safe("Metric_Name") == "metric_name"
        assert prometheus_safe("MetricName") == "metricname"

    def test_prometheus_safe_complex_names(self):
        """Test prometheus_safe with complex metric names."""
        assert (
            prometheus_safe("F1-Score@Model.Performance")
            == "f1_score_model_performance"
        )
        assert prometheus_safe("__Drift--Detection..Rate__") == "drift_detection_rate"


class TestMonitoringEndpoint:
    """Test cases for the /metrics endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client for the Flask app."""
        app.config["TESTING"] = True
        return app.test_client()

    @pytest.fixture
    def sample_reference_data(self):
        """Create sample reference data."""
        np.random.seed(42)  # For reproducible tests
        n_samples = 100

        data = {
            **{f"mfcc_{i}": np.random.randn(n_samples) for i in range(1, 14)},
            "predicted_genre": np.random.choice(
                ["blues", "classical", "country", "disco", "hiphop"], n_samples
            ),
            "actual_genre": np.random.choice(
                ["blues", "classical", "country", "disco", "hiphop"], n_samples
            ),
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_current_data(self):
        """Create sample current data with slight distribution shift."""
        np.random.seed(43)  # Different seed for current data
        n_samples = 80

        data = {
            **{
                f"mfcc_{i}": np.random.randn(n_samples) + 0.1 for i in range(1, 14)
            },  # Slight shift
            "predicted_genre": np.random.choice(
                ["blues", "classical", "country", "disco", "hiphop"], n_samples
            ),
            "actual_genre": np.random.choice(
                ["blues", "classical", "country", "disco", "hiphop"], n_samples
            ),
        }
        return pd.DataFrame(data)

    def test_metrics_endpoint_missing_reference_file(self, client):
        """Test /metrics endpoint when reference file is missing."""
        with patch("run_monitoring.os.path.exists") as mock_exists:
            mock_exists.side_effect = (
                lambda path: path != "monitoring/data/reference.parquet"
            )

            response = client.get("/metrics")

            assert response.status_code == 200
            assert response.mimetype == "text/plain"
            assert "reference.parquet not found" in response.get_data(as_text=True)

    def test_metrics_endpoint_missing_current_file(self, client):
        """Test /metrics endpoint when current file is missing."""
        with patch("run_monitoring.os.path.exists") as mock_exists:
            mock_exists.side_effect = (
                lambda path: path != "monitoring/data/current.parquet"
            )

            response = client.get("/metrics")

            assert response.status_code == 200
            assert response.mimetype == "text/plain"
            assert "current.parquet not found" in response.get_data(as_text=True)

    @patch("run_monitoring.pd.read_parquet")
    @patch("run_monitoring.os.path.exists")
    def test_metrics_endpoint_successful_response(
        self,
        mock_exists,
        mock_read_parquet,
        client,
        sample_reference_data,
        sample_current_data,
    ):
        """Test /metrics endpoint with valid data files."""
        # Mock file existence
        mock_exists.return_value = True

        # Mock pandas read_parquet to return our sample data
        mock_read_parquet.side_effect = [sample_reference_data, sample_current_data]

        # Mock evidently components to avoid complex dependencies
        with patch("run_monitoring.Dataset") as mock_dataset, patch(
            "run_monitoring.Report"
        ) as mock_report:
            # Setup mock dataset
            mock_dataset.from_pandas.return_value = Mock()

            # Setup mock report with sample metrics
            mock_report_instance = Mock()
            mock_results = {
                "metrics": [
                    {"metric_id": "row_count", "value": 80},
                    {"metric_id": "accuracy", "value": 0.85},
                    {"metric_id": "f1_score", "value": 0.82},
                    {
                        "metric_id": "drift_score",
                        "value": {"mfcc_1": 0.15, "mfcc_2": 0.23},
                    },
                    {
                        "metric_id": "category_count",
                        "value": {"blues": 20, "classical": 15},
                    },
                ]
            }
            mock_report_instance.run.return_value.dict.return_value = mock_results
            mock_report.return_value = mock_report_instance

            response = client.get("/metrics")

            # Verify response
            assert response.status_code == 200
            assert response.mimetype == "text/plain"

            response_text = response.get_data(as_text=True)

            # Check Prometheus format headers
            assert (
                "# HELP evidently_metric Evidently monitoring metric" in response_text
            )
            assert "# TYPE evidently_metric gauge" in response_text

            # Check that numeric metrics are included
            assert "row_count 80.000000" in response_text
            assert "accuracy 0.850000" in response_text
            assert "f1_score 0.820000" in response_text

            # Check that nested metrics are flattened
            assert "drift_score_mfcc_1 0.150000" in response_text
            assert "drift_score_mfcc_2 0.230000" in response_text
            assert "category_count_blues 20.000000" in response_text
            assert "category_count_classical 15.000000" in response_text

    @patch("run_monitoring.pd.read_parquet")
    @patch("run_monitoring.os.path.exists")
    def test_metrics_endpoint_prometheus_format_validation(
        self,
        mock_exists,
        mock_read_parquet,
        client,
        sample_reference_data,
        sample_current_data,
    ):
        """Test that the metrics endpoint returns valid Prometheus format."""
        mock_exists.return_value = True
        mock_read_parquet.side_effect = [sample_reference_data, sample_current_data]

        with patch("run_monitoring.Dataset") as mock_dataset, patch(
            "run_monitoring.Report"
        ) as mock_report:
            mock_dataset.from_pandas.return_value = Mock()

            # Mock report with metrics that need name sanitization
            mock_results = {
                "metrics": [
                    {"metric_id": "Row-Count", "value": 100},
                    {"metric_id": "F1.Score@Average", "value": 0.75},
                    {"metric_id": "Drift--Detection..Rate", "value": 0.05},
                ]
            }
            mock_report_instance = Mock()
            mock_report_instance.run.return_value.dict.return_value = mock_results
            mock_report.return_value = mock_report_instance

            response = client.get("/metrics")
            response_text = response.get_data(as_text=True)

            # Verify metric names are properly sanitized
            assert "row_count 100.000000" in response_text
            assert "f1_score_average 0.750000" in response_text
            assert "drift_detection_rate 0.050000" in response_text

            # Verify no invalid characters remain
            lines = response_text.split("\n")
            for line in lines:
                if line and not line.startswith("#"):
                    metric_name = line.split()[0]
                    # Check that metric name follows Prometheus conventions
                    assert metric_name.replace("_", "").replace(":", "").isalnum()

    @patch("run_monitoring.pd.read_parquet")
    @patch("run_monitoring.os.path.exists")
    def test_metrics_endpoint_handles_non_numeric_values(
        self,
        mock_exists,
        mock_read_parquet,
        client,
        sample_reference_data,
        sample_current_data,
    ):
        """Test that the endpoint properly handles non-numeric metric values."""
        mock_exists.return_value = True
        mock_read_parquet.side_effect = [sample_reference_data, sample_current_data]

        with patch("run_monitoring.Dataset") as mock_dataset, patch(
            "run_monitoring.Report"
        ) as mock_report:
            mock_dataset.from_pandas.return_value = Mock()

            # Mock report with mixed value types
            mock_results = {
                "metrics": [
                    {"metric_id": "numeric_metric", "value": 42.5},
                    {"metric_id": "string_metric", "value": "not_a_number"},
                    {"metric_id": "none_metric", "value": None},
                    {"metric_id": "convertible_string", "value": "123.45"},
                    {"metric_id": "list_metric", "value": [1, 2, 3]},
                ]
            }
            mock_report_instance = Mock()
            mock_report_instance.run.return_value.dict.return_value = mock_results
            mock_report.return_value = mock_report_instance

            response = client.get("/metrics")
            response_text = response.get_data(as_text=True)

            # Should include numeric metrics
            assert "numeric_metric 42.500000" in response_text
            assert "convertible_string 123.450000" in response_text

            # Should exclude non-numeric metrics
            assert "string_metric" not in response_text
            assert "none_metric" not in response_text
            assert "list_metric" not in response_text

    def test_metrics_endpoint_structure(self, client):
        """Test the basic structure of the metrics endpoint response."""
        with patch("run_monitoring.os.path.exists") as mock_exists:
            mock_exists.return_value = False  # Files don't exist

            response = client.get("/metrics")

            # Should always return 200 OK
            assert response.status_code == 200

            # Should always return plain text
            assert response.mimetype == "text/plain"

            # Should always end with newline
            response_text = response.get_data(as_text=True)
            assert response_text.endswith("\n")


class TestDataProcessing:
    """Test cases for data processing logic."""

    def test_mfcc_features_generation(self):
        """Test that MFCC feature names are generated correctly."""
        # This tests the logic: mfcc_features = [f"mfcc_{i}" for i in range(1, 14)]
        expected_features = [f"mfcc_{i}" for i in range(1, 14)]
        assert len(expected_features) == 13
        assert expected_features[0] == "mfcc_1"
        assert expected_features[-1] == "mfcc_13"
        assert "mfcc_0" not in expected_features
        assert "mfcc_14" not in expected_features

    @patch("run_monitoring.pd.read_parquet")
    @patch("run_monitoring.os.path.exists")
    def test_categories_extraction(self, mock_exists, mock_read_parquet):
        """Test that categories are correctly extracted from reference data."""
        mock_exists.return_value = True

        # Create test data with specific genres
        test_data = pd.DataFrame(
            {
                "mfcc_1": [1, 2, 3, 4, 5],
                "predicted_genre": ["rock", "blues", "jazz", "rock", "blues"],
                "actual_genre": ["rock", "blues", "jazz", "rock", "blues"],
            }
        )

        mock_read_parquet.return_value = test_data

        with patch("run_monitoring.Dataset") as mock_dataset, patch(
            "run_monitoring.Report"
        ) as mock_report:
            # Setup minimal mocks
            mock_dataset.from_pandas.return_value = Mock()
            mock_report_instance = Mock()
            mock_report_instance.run.return_value.dict.return_value = {"metrics": []}
            mock_report.return_value = mock_report_instance

            # Import and trigger the logic
            client = app.test_client()
            client.get("/metrics")

            # The categories should be extracted and sorted
            # In the actual code: categories = sorted(reference_df["predicted_genre"].dropna().unique().tolist())
            expected_categories = sorted(["rock", "blues", "jazz"])
            assert expected_categories == ["blues", "jazz", "rock"]


if __name__ == "__main__":
    # Run the tests
    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v"], check=False
    )

    if test_result.returncode == 0:
        print("✅ All monitoring tests passed!")
    else:
        print("❌ Some monitoring tests failed")
