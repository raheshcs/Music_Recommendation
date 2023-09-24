from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
from keras.models import load_model
import webbrowser
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random

client_id = '2bc1c233bcdc42cebada5043e3fdc699'
client_secret = 'e915d339808f457e8bba429061fe7625'

# Initialize the Spotify API client
client_credentials_manager = SpotifyClientCredentials(client_id, client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

info = {}

haarcascade = "haarcascade_frontalface_default.xml"
label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
print("+"*50, "loadin gmmodel")
model = load_model('model.h5')
cascade = cv2.CascadeClassifier(haarcascade)
@app.route('/')
def load():
	return redirect(url_for('index'))

@app.route('/index')
def index():
	found = False

	cap = cv2.VideoCapture(0)
	while not(found):
		_, frm = cap.read()
		gray = cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)

		faces = cascade.detectMultiScale(gray, 1.4, 1)

		for x,y,w,h in faces:
			found = True
			roi = gray[y:y+h, x:x+w]
			cv2.imwrite("static/face.jpg", roi)

	roi = cv2.resize(roi, (48,48))

	roi = roi/255.0
	
	roi = np.reshape(roi, (1,48,48,1))

	prediction = model.predict(roi)

	prediction = np.argmax(prediction)
	prediction = label_map[prediction]

	cap.release()

	

	moods = {
    "Happy": "spotify:playlist:37i9dQZF1DX2x1COalpsUi",
    "Sad": "spotify:playlist:6ybCs40CUSVISYNRoqtQAZ",
    "Neutral": "spotify:playlist:4LofyaDGT8cMr6KehRhMow",
    "Suprise": "spotify:playlist:37i9dQZF1DWZS4GhkDZq7c",
    "Fear":"spotify:playlist:3Ugn19T02woShoErw9q9TT",
    "Anger":"spotify:playlist:3yFeetVilTGS5fC3JhOeA7",
    "Disgust":"spotify:playlist:4WLYxh12Apkek01oogU89f"
	}

	playlist_uri = moods[prediction]
	playlist = sp.playlist(playlist_uri)
	track = random.choice(playlist['tracks']['items'])['track']

	
	#webbrowser.open(track['uri'])
	splink=track['uri']
	ytlink=track['name']
	ytlink  = f"https://www.youtube.com/results?search_query={track['name']}+song"
	webbrowser.open(splink)
	return render_template("index.html", data=prediction,splink=splink,ytlink=ytlink)


if __name__ == "__main__":
	app.run(debug=True)