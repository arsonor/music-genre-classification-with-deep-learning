"""
Genre prediction service for music classification using MLflow models.
"""
import os

import numpy as np
import mlflow
import librosa
from mlflow import MlflowClient

SAMPLE_RATE = 22050
TEST_DURATION = 3  # measured in seconds
SAMPLES_TO_CONSIDER = SAMPLE_RATE * TEST_DURATION
MODEL_NAME = "music_genre_tf_model"


class _Genre_Prediction_Service:
    """Singleton class for genre prediction inference with trained models.

    Attributes:
        model: Trained model for genre prediction
        _mapping: List of genre labels corresponding to model output indices
        _instance: Singleton instance of the class
    """

    model = None
    _mapping = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock",
    ]
    _instance = None

    def predict(self, audio_file_path):
        """
        Predict the genre of an audio file.

        Args:
            audio_file_path (str): Path to audio file to predict

        Returns:
            str: Genre predicted by the model
        """
        # extract MFCC
        mfccs = self.preprocess(audio_file_path)

        # we need a 4-dim array to feed to the model for prediction:
        # (# samples, # time steps, # coefficients, 1)
        mfccs = mfccs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(mfccs)
        predicted_index = np.argmax(predictions)
        predicted_genre = self._mapping[predicted_index]
        return predicted_genre

    def preprocess(self, audio_file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """
        Extract MFCCs from audio file.

        Args:
            audio_file_path (str): Path of audio file
            num_mfcc (int): Number of coefficients to extract
            n_fft (int): Interval we consider to apply FFT. Measured in # of samples
            hop_length (int): Sliding window for FFT. Measured in # of samples

        Returns:
            numpy.ndarray: 2-dim array with MFCC data of shape (# time steps, # coefficients)

        Raises:
            ValueError: If audio signal is too short for processing
        """
        # load audio file
        signal, sample_rate = librosa.load(audio_file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=signal,
                sr=sample_rate,
                n_mfcc=num_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            return mfccs.T

        raise ValueError("Audio signal too short for processing.")

    def extract_mean_mfcc(self, audio_file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        Extract and return the mean MFCC vector from the audio file.

        Args:
            audio_file_path (str): Path to audio file
            n_mfcc (int): Number of MFCC coefficients to extract
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames

        Returns:
            numpy.ndarray: Mean MFCC vector of shape (n_mfcc,)
        """
        signal, sample_rate = librosa.load(audio_file_path, sr=22050)
        mfcc = librosa.feature.mfcc(
            y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean

    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of the genre prediction service.

        Returns:
            _Genre_Prediction_Service: The singleton instance
        """
        return cls._instance

    @classmethod
    def set_instance(cls, instance):
        """
        Set the singleton instance (used by factory function).

        Args:
            instance (_Genre_Prediction_Service): The instance to set
        """
        cls._instance = instance


def Genre_Prediction_Service():  # pylint: disable=invalid-name
    """
    Factory function for Genre_Prediction_Service class.

    This function implements the singleton pattern to ensure only one instance
    of the genre prediction service exists and loads the model only once.

    Returns:
        _Genre_Prediction_Service: The singleton instance of the service
    """
    # ensure an instance is created only the first time the factory function is called
    if _Genre_Prediction_Service.get_instance() is None:
        print("Loading latest MLflow model from registry...")
        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest_version = max(versions, key=lambda v: int(v.version))
        model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
        model = mlflow.keras.load_model(model_uri)

        print(f"Model loaded: version {latest_version.version}")

        # Create new instance and set the model
        instance = _Genre_Prediction_Service()
        instance.model = model
        _Genre_Prediction_Service.set_instance(instance)

    return _Genre_Prediction_Service.get_instance()


def main():
    """
    Main function to demonstrate the genre prediction service.
    """
    # create 2 instances of the genre prediction service
    gps = Genre_Prediction_Service()
    gps1 = Genre_Prediction_Service()

    # check that different instances of the genre prediction service point back to the same object (singleton)
    assert gps is gps1

    # make predictions
    test_file_paths = [
        "../test/blues.00000.wav",
        "../test/classical.00000.wav",
        "../test/country.00000.wav",
        "../test/disco.00000.wav",
        "../test/hiphop.00000.wav",
        "../test/jazz.00000.wav",
        "../test/metal.00000.wav",
        "../test/pop.00000.wav",
        "../test/reggae.00000.wav",
        "../test/rock.00000.wav",
    ]

    for test_file_path in test_file_paths:
        if os.path.exists(test_file_path):
            genre = gps.predict(test_file_path)
            print(f"File: {test_file_path}, Predicted Genre: {genre}")
        else:
            print(f"File not found: {test_file_path}")


if __name__ == "__main__":
    main()
