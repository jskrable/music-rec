# Music Recommendation Service

This project is a music recommendation service. It draws from the [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/), and attempts to recommend songs based on a list of liked tracks from the user. The recommendation engine is a deep learning artificial neural network implemented with [Keras](https://keras.io/) and [TensorFlow](https://github.com/tensorflow/tensorflow). The model is run using [Flask](https://github.com/pallets/flask/) to host middleware APIs, and the frontend is based on [Skeleton](http://getskeleton.com/).

Install dependencies with 
```
pip install -r requirements.txt
```



To standup web app, first run 
```
cd frontend
python -m http.server
```
This will serve the static site html pages

In another terminal window, run
```
python api/api.py
```
This will host the middleware api endpoints on http://localhost:5000

Then visit the static site at http://localhost:8000 and try submitting some songs for recommendation.
