import numpy as np
import imutils
import pickle
import cv2
import os
import time
import threading
from PIL import ImageGrab 
from face_detect import face_detect
from recognizer import recognizer
from train_model import train_model

recognizer = recognizer()
face_detect = face_detect()
train_model = train_model()
train_model.train()
print("Start")

name = ""
nameOutputFlag = 0
learningFlag = 0
unknownFlag = 0
nameinput = ""
UnknownImage = ""

def Thread_recognize():
	global name
	global nameOutputFlag
	global UnknownImage
	global unknownFlag
	video_capture = cv2.VideoCapture(0)
	while True:
		ret, frame = video_capture.read()
		#image=ImageGrab.grab(bbox=(0, 200, 1200, 800))
		#printScreen = np.array(image)
		#frame = cv2.cvtColor(printScreen, cv2.COLOR_BGR2RGB)
		frame= imutils.resize(frame, width=500)
		embedding, detected_faces = face_detect.get_Featurepoint(frame)
		for i, face_rect in enumerate(detected_faces):
			left, top, right, bottom = face_rect
			if abs(right - left) < 100:
				break
			name, probability = recognizer.recognize(embedding[i],0.7)
			nameOutputFlag = 1
			if name == "Unknown":
				if unknownFlag != 1:
					UnknownImage = frame
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
				0.75, (0, 255, 0), 1)
			break #One Person
		cv2.imshow("Image", frame)
		cv2.waitKey(1)
	video_capture.release()

def Thread_learning():
	global learningFlag
	global nameinput
	global UnknownImage
	global unknownFlag
	while(True):
		if unknownFlag == 1:
			frame = UnknownImage
			cv2.imshow("Unknown", frame)
			cv2.waitKey(1)
		if learningFlag == 1:
			for i in range(2):
				train_model.collect_embedding(frame, nameinput)
				train_model.embedding()
				train_model.train()
				cv2.destroyWindow("Unknown")
				recognizer.reload()
				learningFlag = 0
				unknownFlag = 0
			

t1 = threading.Thread(target=Thread_recognize)
t1.daemon = True
t1.start()

t2 = threading.Thread(target=Thread_learning)
t2.daemon = True
t2.start()

while(True):
	command = input("exit, recog, delete\n명령어 입력: ")
	if command == "exit":
		print("종료")
		break
	elif command == "recog":
		if nameOutputFlag == 1:
			os.system('cls')
			print(name)
			if name == "Unknown":
				unknownFlag = 1
				nameinput = input("이름 입력 : ")
				if nameinput == "q":
					unknownFlag = 0
					cv2.destroyWindow("Unknown")
					continue
				learningFlag = 1
			time.sleep(1)
			nameOutputFlag = 0
	elif command == "delete":
		deleteName = input("삭제할 이름 입력: ")
		train_model.delete(deleteName)
		recognizer.reload()
		os.system('cls')
	else:
		continue

cv2.destroyAllWindows()