"""
Prefect flow for music genre classification pipeline.

This module defines the orchestration workflow for training the music genre
classification model using Prefect for workflow management.
"""
from prefect import flow, task
from classifier.pipeline.train import train_and_log_model
from classifier.pipeline.features import extract_features, download_and_validate_data


@task
def train_task(file_path):
    """
    Prefect task wrapper for model training.

    Args:
        file_path (str): Path to the training data file

    Returns:
        dict: Training metrics and results from the model training process
    """
    # Return training metrics for potential use in downstream tasks
    return train_and_log_model(file_path)


@task
def validate_data_task(data_path):
    """
    Prefect task to validate and extract features from data.

    Args:
        data_path (str): Path to the data file

    Returns:
        tuple: Features (X) and labels (y) extracted from the data
    """
    features, labels = extract_features(data_path)
    print(
        f"Data validation complete: {features.shape[0]} samples, {features.shape[1:]} feature shape"
    )
    print(f"Labels shape: {labels.shape}")
    return features, labels


@flow(name="music-genre-pipeline")
def music_genre_pipeline():
    """
    Main Prefect flow for the music genre classification pipeline.

    This flow orchestrates the complete machine learning pipeline:
    1. Download and validate training data
    2. Extract features from the data
    3. Train the model using the processed data

    Returns:
        dict: Results from the training task
    """
    # Step 1: Download and validate data
    data_path = download_and_validate_data()

    # Step 2: Extract and validate features
    features, labels = validate_data_task(data_path)  # pylint: disable=unused-variable

    # Step 3: Train the model
    # Note: The training function loads data internally, so we pass the data_path
    # The features and labels are used for validation/logging purposes
    training_results = train_task(data_path)

    return training_results


def main():
    """
    Main function to execute the music genre classification pipeline.
    """
    print("Starting music genre classification pipeline...")
    results = music_genre_pipeline()
    print(f"Pipeline completed successfully: {results}")


if __name__ == "__main__":
    main()
