#!/usr/bin/env python3
# coding: utf-8
"""
api.py
04-14-19
jack skrable
"""

# USAGE
# Start the server:
# 	python api.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import numpy as np
import pandas as pd
import flask


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global model
	model = ResNet50(weights="imagenet")

def load_lookups():
	global lookupDF
	lookupDF = pd.read_hdf('./frontend/data/lookup.h5', 'df')


@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			preds = model.predict(image)
			results = imagenet_utils.decode_predictions(preds)
			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			for (imagenetID, label, prob) in results[0]:
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)


@app.route("/lookup", methods=["GET"])
# @cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def lookup():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "GET":
		
		# Get lookup data here		
		data['entity'] = lookupDF.to_dict('records')

		# indicate that the request was a success
		data["success"] = True

	# JSONify data for response
	response = flask.jsonify(data)
	# Allow CORS
	response.headers.add('Access-Control-Allow-Origin', '*')

	return response

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(" * Starting Flask server and loading Keras model...")
	print(" * Please wait until server has fully started")
	# Load my model 
	# load_model()
	# Get songs for lookup function
	load_lookups()

	print(' * Server is active')
	# Run app
	app.run(host='0.0.0.0', port=5001)