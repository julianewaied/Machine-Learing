import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from time import sleep

data_path = "C:/Users/WIN10PRO/Desktop/My Stuff/University/BSC/Machine Learning/Machine-Learing/HW2/q3/exams.csv"
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
	labels = np.array([x[2] for x in data])
	if(change):
		labels[labels == 0] = -1
	return points,labels

class logistic_regression:
	def __init__(self,learning_rate = 1,convergance_constant = 0.1):
		self.etta = learning_rate
		self.convergance = convergance_constant
	def calc_grad(self,samples,lbl):
		vec_grad = np.vectorize(self.single_grad, signature='(n),()->(n)')
		grads = vec_grad(samples,lbl)
		gradient = np.sum(grads,axis = 0)
		return gradient
	def single_grad(self, x,y):
		# to prevent overflow
		g = self.g(x)
		g = np.clip(g, -500, 500)
		scalar = (-y)/(1+np.exp(y*g))
		answer = scalar * x
		return answer
	def train(self,samples,labels):
		# iterate with gradient descent
		# w is initially 0 vector
		self.w = np.array([0 for i in range(0,len(samples[0]))])
		i=0
		prev = 0
		curr =np.linalg.norm(self.calc_grad(samples,labels))
		while(i<9000 and curr>self.convergance):
			gradient = self.calc_grad(samples,labels)
			norm = np.linalg.norm(gradient)
			if(norm<curr):
				if(norm<0.98*curr and self.etta>0.01):
					self.etta = self.etta/2
				prev = curr
				curr = norm
			self.w = self.w - self.etta * gradient
			i+=1
		return self.w
	def count_pred(self,sample,label):
		if(self.classify(sample) == label):
			return 1
		return 0
	def test_model(self,samples,labels):
		count = 0
		pred = np.vectorize(self.count_pred,signature='(n),()->()')
		predictions = pred(samples,labels)
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
	best_w = np.array([0 for i in range(0,len(samples[0]))])
	w = np.array([0 for i in range(0,len(samples[0]))])
	for i in range(0,epoch_num):
		samp_train, samp_test, lbl_train, lbl_test = train_test_split(samples, labels, test_size=0.1, random_state=42)
		curr_w = model.train(samp_train,lbl_train)
		w =  w + curr_w
		acc = model.test_model(samp_test,lbl_test)
		if(acc>best_acc):
			best_w = curr_w
			best_acc = acc
		epoch_sum = epoch_sum + acc
	return (epoch_sum/epoch_num), best_w



data = load(data_path)
# Accuracy is 80%
model = logistic_regression(convergance_constant=0.357,learning_rate=3.5)
accuracy, w = evaluate(model,data)
print(f"Avg test accuracy: {accuracy * 100}%")

