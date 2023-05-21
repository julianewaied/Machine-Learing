import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

data_path = "exams.csv"
def plot(data, labels, w):
	fig, ax = plt.subplots()

	c0 = data[labels == 0]
	c1 = data[labels == 1]

	ax.scatter(c0[:,0], c0[:,1], c='red')
	ax.scatter(c1[:,0], c1[:,1], c='blue')
	
	a, b, c = w
	m = -a / b
	b = -c / b

	x = np.arange(np.min(data[:,0]), np.max(data[:,0]), 0.1)
	y = m * x + b
	plt.plot(x, y)

	plt.show()

def load(path):
	data = pd.read_csv(path,header=None)
	return data

def preprocess(data,change=True):
	points = np.array([np.array([x[0],x[1],1]) for x in data])
	labels = data[:,2]
	if(change):
		labels[labels == 0] = -1
	return points,labels

class logistic_regression:
	def __init__(self,learning_rate = 1,convergance_constant = 0.1):
		self.etta = learning_rate
		self.convergance = convergance_constant
	def calc_grad(self,samples,lbl):
		grads = np.vectorize(self.single_grad, signature='(n),()->(n)')(samples,lbl)
		gradient = np.mean(grads,axis = 0)
		return gradient
	def single_grad(self, x,y):
		g = self.g(x)
		scalar = (-y)/(1+np.exp(y*g))
		answer = scalar * x
		return answer
	def train(self,samples,labels):
		# iterate with gradient descent
		self.w = np.full(samples.shape[1],0.5)
		i=0
		prev = 0
		curr =np.linalg.norm(self.calc_grad(samples,labels))
		# to ensure no infinite loops for futuristic training data
		while(curr>self.convergance and i<28000):
			gradient = self.calc_grad(samples,labels)
			curr = np.linalg.norm(gradient)
			self.w = self.w - self.etta * gradient
			i+=1
		return self.w
	def count_pred(self,sample,label):
		if(self.classify(sample) == label):
			return 1
		return 0
	def test_model(self,samples,labels):
		predictions = np.vectorize(self.count_pred,signature='(n),()->()')(samples,labels)
		return np.mean(predictions)
		
	def classify(self,x):
		return np.sign(self.g(x))
	def g(self,x):
		# w includes w0 - wd
		return np.dot(self.w,x)
	def setW(self,w):
		self.w = w

def evaluate(model,data):
	epoch_num = 4
	epoch_sum = 0
	best_acc = 0
	data = data.to_numpy()
	samples,labels = preprocess(data)
	best_w = np.zeros(samples.shape[1])
	w = np.zeros(samples.shape[1])
	for i in range(0,epoch_num):
		samp_train, samp_test, lbl_train, lbl_test = train_test_split(samples, labels, test_size=0.1, random_state=42)
		curr_w = model.train(samp_train,lbl_train)
		acc = model.test_model(samp_test,lbl_test)
		if(acc>best_acc):
			best_w = curr_w
			best_acc = acc
		epoch_sum = epoch_sum + acc
	return (epoch_sum/epoch_num), best_w


data = load(data_path)
# Accuracy is 90%
model = logistic_regression(convergance_constant=1,learning_rate=6.352631578947369)
accuracy, w = evaluate(model,data)
print(f"Avg test accuracy: {accuracy * 100}%")
# data = data.to_numpy()
# plot(data[:,0:2],data[:,2],w)
