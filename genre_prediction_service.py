import librosa
import tensorflow as tf
import numpy as np

SAVED_MODEL_PATH = "model.keras"
SAMPLE_RATE = 22050
TEST_DURATION = 3 # measured in seconds
SAMPLES_TO_CONSIDER = SAMPLE_RATE * TEST_DURATION

class _Genre_Prediction_Service:
    """Singleton class for genre prediction inference with trained models.

    :param model: Trained model
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
        "rock"
    ]
    _instance = None


    def predict(self, file_path):
        """

        :param file_path (str): Path to audio file to predict
        :return predicted_genre (str): Genre predicted by the model
        """

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_genre = self._mapping[predicted_index]
        return predicted_genre


    def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCCs from audio file.

        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples

        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        """

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T


def Genre_Prediction_Service():
    """Factory function for Genre_Prediction_Service class.

    :return _Genre_Prediction_Service._instance (_Genre_Prediction_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Genre_Prediction_Service._instance is None:
        _Genre_Prediction_Service._instance = _Genre_Prediction_Service()
        _Genre_Prediction_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Genre_Prediction_Service._instance




if __name__ == "__main__":

    # create 2 instances of the genre prediction service
    gps = Genre_Prediction_Service()
    gps1 = Genre_Prediction_Service()

    # check that different instances of the genre prediction service point back to the same object (singleton)
    assert gps is gps1

    # make predictions
    file_paths = [
        "test/blues.00000.wav",
        "test/classical.00000.wav",
        "test/country.00000.wav",
        "test/disco.00000.wav",
        "test/hiphop.00000.wav",
        "test/jazz.00000.wav",
        "test/metal.00000.wav",
        "test/pop.00000.wav",
        "test/reggae.00000.wav",
        "test/rock.00000.wav"
    ]

    for file_path in file_paths:
        genre = gps.predict(file_path)
    
        print(f"File: {file_path}, Predicted Genre: {genre}")
