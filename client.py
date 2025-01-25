import requests

# server url
URL = "http://127.0.0.1:80/predict"


# audio file we'd like to send for predicting genre
FILE_PATH = "test/blues.00000.wav"


if __name__ == "__main__":

    # open files
    file = open(FILE_PATH, "rb")

    # package stuff to send and perform POST request
    values = {"file": (FILE_PATH, file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print("Predicted genre: {}".format(data["genre"]))