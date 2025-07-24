import random
import os
from flask import Flask, request, jsonify
from genre_prediction_service import Genre_Prediction_Service


# instantiate flask app
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
	"""Endpoint to predict genre

    :return (json): This endpoint returns a json file with the following format:
        {
            "genre": "blues"
        }
	"""

	# get file from POST request and save it
	audio_file = request.files["file"]
	file_name = str(random.randint(0, 100000))
	audio_file.save(file_name)

	# instantiate genre prediction service singleton and get prediction
	gps = Genre_Prediction_Service()
	predicted_genre = gps.predict(file_name)

	# we don't need the audio file any more - let's delete it!
	os.remove(file_name)

	# send back result as a json file
	result = {"genre": predicted_genre}
	return jsonify(result)


if __name__ == "__main__":
    print("Starting genre prediction service...")
    app.run(debug=False)