import numpy as np
import matplotlib.pyplot as plt

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

class logistic_regression:
	def __init__(self):
		w=[]
# accuracy = 0
# print(f"Avg test accuracy: {accuracy * 100}%")	



#-----------------------------------------