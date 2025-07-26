import requests

# Server URL
URL = "http://127.0.0.1:80/predict"

# Path to the audio file to send
FILE_PATH = "test/metal.00000.wav"

# Optional: true label (ground truth)
TRUE_GENRE = "metal"

if __name__ == "__main__":
    # Open the audio file
    with open(FILE_PATH, "rb") as file:
        # 'files' = multipart form with the audio file
        files = {"file": (FILE_PATH, file, "audio/wav")}

        # 'data' = form field with the true label
        data = {"actual_genre": TRUE_GENRE}

        # Send POST request
        response = requests.post(URL, files=files, data=data)

    # Parse response
    if response.status_code == 200:
        result = response.json()
        print("Predicted genre: {}".format(result.get("predicted_genre", "Unknown")))
    else:
        print("Request failed with status code:", response.status_code)
        print(response.text)
