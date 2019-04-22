# Music Recommendation Service

This project is a music recommendation service. It draws from the [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/), and attempts to recommend songs based on a list of liked tracks from the user. The recommendation engine is a deep learning artificial neural network implemented with [Keras](https://keras.io/) and [TensorFlow](https://github.com/tensorflow/tensorflow). The model is run using [Flask](https://github.com/pallets/flask/) to host middleware APIs, and the frontend is based on [Skeleton](http://getskeleton.com/).

Install dependencies with 
```
pip install -r requirements.txt
```

To get the dataset, you can either run ```setup.sh``` or the following commands
```
mkdir data && cd data
wget http://static.echonest.com/millionsongsubset_full.tar.gz
tar -xvzf millionsongsubset_full.tar.gz
```
Alternatively, visit the Million Song Dataset's site to download the file and unzip it manually.

Training the neural network may take some trial and error based on the hardware available. Running 
```./lib/main.py``` will attempt to train the network using the full dataset and default hyperparameters.

Once the network is trained and a workable model is created, you can run the web app that recommends songs. First, move your best working model (and all the associated files) into ```./model/working/```. Then, to standup web app, first run 
```
cd frontend
python -m http.server
```
This will serve the static site html pages

In another terminal window, run
```
python src/api.py
```
This will stand up the middleware api endpoints.

Then visit the static site at http://localhost:8000 and try submitting some songs for recommendation.
