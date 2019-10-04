from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from face_detect import face_detect
import pickle

knownEmbeddings = []
knownName = []
class train_model:
	def __init__(self):
		self.recognizer = SVC(C=10.0, kernel="rbf",gamma='auto', probability=True)
		self.le = LabelEncoder()
		#self.knownEmbeddings = []
		#self.knownName = []
		self.face_detect = face_detect()
		
	def embedding(self):
		global knownEmbeddings
		global knownName
		data = pickle.loads(open("embeddings.pickle", "rb").read())
		for i in range(len(data["embeddings"])):
			knownEmbeddings.append(data["embeddings"][i])
			knownName.append(data["names"][i])
		data = {"embeddings": knownEmbeddings, "names": knownName}
		knownEmbeddings = []
		knownName = []
		f = open("embeddings.pickle", "wb")
		f.write(pickle.dumps(data))
		f.close()

	def delete(self, name):
		targetName = []
		targetEmbeddings = []
		data = pickle.loads(open("embeddings.pickle", "rb").read())
		for i in range(len(data["names"])):
			if data["names"][i] == name:
				continue
			targetEmbeddings.append(data["embeddings"][i])
			targetName.append(data["names"][i])
		data = {"embeddings": targetEmbeddings, "names": targetName}
		f = open("embeddings.pickle", "wb")
		f.write(pickle.dumps(data))
		f.close()
		self.train()

	def train(self):
		data = pickle.loads(open("embeddings.pickle", "rb").read())
		labels = self.le.fit_transform(data["names"])
		self.recognizer.fit(data["embeddings"], labels)
		# write the actual face recognition model to disk
		f = open("recognize_pickle/recognizer.pickle", "wb")
		f.write(pickle.dumps(self.recognizer))
		f.close()
		# write the label encoder to disk
		f = open("recognize_pickle/le.pickle", "wb")
		f.write(pickle.dumps(self.le))
		f.close()
		
	def collect_embedding(self, image, name):
		global knownEmbeddings
		global knownName
		embeddings, count = self.face_detect.create_Feature(image)
		for i in range(5):
			for k in range(len(embeddings)):
				knownEmbeddings.append(embeddings[k])
				knownName.append(name)
		
