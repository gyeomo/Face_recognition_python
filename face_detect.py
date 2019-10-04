import dlib
from front_detect import front_detect
import cv2
import openface

class face_detect:
	def __init__(self):
		self.face_detector = dlib.get_frontal_face_detector()
		self.front_detector = front_detect()
		self.embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")
		self.predictor_model = "shape_predictor_68_face_landmarks.dat"
		self.face_aligner = openface.AlignDlib(self.predictor_model)
		
	def rect_to_bb(self, rect):
		x = rect.left()
		y = rect.top()
		w = rect.right() - x
		h = rect.bottom() - y
		return (x, y, w, h)

	def detect(self, image):
		face_rectangle = []
		detected_faces = self.face_detector(image, 1)
		for i, face_rect in enumerate(detected_faces):
			if self.front_detector.detect(image, face_rect):
				(x, y, w, h) = self.rect_to_bb(face_rect)
				left, top, right, bottom = x, y, x+w,y+h
				face_rectangle.append([left, top, right, bottom])
		return face_rectangle
		
	def get_Featurepoint(self, image):
		vec = []
		face_rectangle = []
		detected_faces = self.face_detector(image, 1)
		for i, face_rect in enumerate(detected_faces):
			if self.front_detector.detect(image, face_rect):
				(x, y, w, h) = self.rect_to_bb(face_rect)
				left, top, right, bottom = x, y, x+w,y+h
				alignedFace = self.face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
				faceBlob = cv2.dnn.blobFromImage(alignedFace, 1.0 / 255,
						(96, 96), (0, 0, 0), swapRB=True, crop=False)
				self.embedder.setInput(faceBlob)
				vec.append(self.embedder.forward())
				face_rectangle.append([left, top, right, bottom])
				break #One Person
		return vec, face_rectangle
				
				
	def create_Feature(self, image):
		vec = []
		count = 0
		images = [image, cv2.flip(image,1)]
		for image_ in images:
			detected_faces = self.face_detector(image_, 1)
			for i, face_rect in enumerate(detected_faces):
				if self.front_detector.detect(image_, face_rect):
					alignedFace = self.face_aligner.align(534, image_, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
					faceBlob = cv2.dnn.blobFromImage(alignedFace, 1.0 / 255,
							(96, 96), (0, 0, 0), swapRB=True, crop=False)
					self.embedder.setInput(faceBlob)
					vec.append(self.embedder.forward().flatten())		
					count+=1
					break #One Person
		return vec, count
				
				
				
				
				
				