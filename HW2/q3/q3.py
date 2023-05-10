import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def test_model(model,data):
	samples, labels = preprocess(data)
	count = 0
	for i in range(0,len(samples)):
		if(model.classify(samples[i])==labels[i]):
			count += 1
	return count/float(len(samples))
		
def load(path):
	data = pd.read_csv(path,header=None)
	train = data.head(int(len(data)*0.9)).to_numpy()
	test = data.tail(len(data)-len(train)).to_numpy()
	return train,test

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
	def train(self,data):
		# iterate with gradient descent
		# w is initially 0 vector
		samples, labels = preprocess(data)
		self.w = np.array([0 for i in range(0,len(samples[0]))])
		i=0
		prev = 0
		curr =np.linalg.norm(self.calc_grad(samples,labels))
		while(curr>self.convergance):
			gradient = self.calc_grad(samples,labels)
			norm = np.linalg.norm(gradient)
			if(norm<curr):
				if(norm<0.98*curr and self.etta>0.0001):
					self.etta = self.etta/2
				prev = curr
				curr = norm
			self.w = self.w - self.etta * gradient
			i+=1
				
		return self.w
	def classify(self,x):
		return np.sign(self.g(x))
	def g(self,x):
		# w includes w0 - wd
		return np.dot(self.w,x)
	def setW(self,w):
		self.w = w



train,test = load(data_path)
model = logistic_regression(convergance_constant=0.357,learning_rate=2)
d,l = preprocess(train,change = False)
# print('training...')
w = model.train(train)
# print('w = ', w)
# train_acc =test_model(model,train)*100
# print('train acc : ',train_acc,'%')
plot(d,l,w)
d,l = preprocess(test,change = False)
plot(d,l,w)
accuracy = test_model(model,test)
print(f"Avg test accuracy: {accuracy * 100}%")	
