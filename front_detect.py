import pickle
import cv2
import numpy as np
import dlib
import os

class front_detect:
	def __init__(self):
		self.predictor_model = "shape_predictor_68_face_landmarks.dat"
		self.face_pose_predictor = dlib.shape_predictor(self.predictor_model)
		self.front_detector = pickle.loads(open("front_detect_pickle/front_detecting.pickle", "rb").read())
		self.target = pickle.loads(open("front_detect_pickle/target.pickle", "rb").read())

	def get_front(self, pose_landmarks):
		out_left = [pose_landmarks.part(0).x, pose_landmarks.part(0).y]
		out_right = [pose_landmarks.part(16).x, pose_landmarks.part(16).y]
		eye_left = [pose_landmarks.part(36).x, pose_landmarks.part(36).y]
		eye_right = [pose_landmarks.part(45).x, pose_landmarks.part(45).y]
		out_bottom = [pose_landmarks.part(8).x, pose_landmarks.part(8).y]
		nose1 = [pose_landmarks.part(30).x, pose_landmarks.part(30).y]
		nose2 = [pose_landmarks.part(27).x, pose_landmarks.part(27).y]
		if (eye_left[0] - nose1[0]) != 0:
			eye_nose_left = abs((eye_left[1] - nose1[1])/ (eye_left[0] - nose1[0]))
		else:
			eye_nose_left = abs((eye_left[1] - nose1[1])/ (0.00001))
		if (eye_right[0] - nose1[0]) != 0:
			eye_nose_right = abs((eye_right[1] - nose1[1])/ (eye_right[0] - nose1[0]))
		else:
			eye_nose_right = abs((eye_right[1] - nose1[1])/ (0.00001))
		nose = abs(nose1[0] - nose2[0])
		return abs(eye_nose_left -eye_nose_right),nose
		
	def detect(self, image, face_rect):
		pose_landmarks = self.face_pose_predictor(image, face_rect)
		value = np.array(self.get_front(pose_landmarks))
		preds = self.front_detector.predict_proba(value.reshape(1,-1))[0]
		j = np.argmax(preds)
		proba = preds[j]
		front = self.target.classes_[j]
		if front == "1":
			return 1
		return 0
	


