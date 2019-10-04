import numpy as np
import pickle
import os


class recognizer:
	def __init__(self):
		self.recognizer = pickle.loads(open("recognize_pickle/recognizer.pickle", "rb").read())
		self.le = pickle.loads(open("recognize_pickle/le.pickle", "rb").read())

	def recognize(self, embedding, threshold):
		name = "Unknown"
		preds = self.recognizer.predict_proba(embedding)[0]
		if np.max(preds)>threshold:
			j = np.argmax(preds)
			proba = preds[j]
			name = self.le.classes_[j]
			
		return name, np.max(preds)
	
	def reload(self):
		self.recognizer = pickle.loads(open("recognize_pickle/recognizer.pickle", "rb").read())
		self.le = pickle.loads(open("recognize_pickle/le.pickle", "rb").read())