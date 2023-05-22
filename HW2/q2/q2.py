﻿import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from math import inf

def plot_cdf(data):
	sorted_data = np.sort(data)[::-1]

	data_cumsum = np.cumsum(sorted_data)
	data_normalized = data_cumsum / data_cumsum[-1]

	# Plot the CDF of eigenvalues
	plt.plot(np.arange(1, len(sorted_data)+1), data_normalized)
	plt.xlabel('Principal Component')
	plt.ylabel('Cumulative Proportion of Variance')
	plt.title('Cumulative Distribution Function of Eigenvalues')
	plt.show()

def load(path):
	data = pd.read_csv(path)
	data = data.to_numpy()
	labels = data[:,0]
	samples = data[:,1:]
	return samples,labels
def isCorrect(model, sample,y):
	if(model.classify(sample)==y):
		return 1
	return 0
def test(model,test_samples,test_labels):
	func = np.vectorize(isCorrect,signature="(),(n),()->()")
	return np.mean(func(model,test_samples,test_labels))
class PCA():

	def __init__(self,data,k):
		self.k = k
		S = (1/data.shape[0]) * np.cov(np.stack(data).transpose())
		eigenvalues,eigenvectors = np.linalg.eigh(S)
		sorted_indices = np.argsort(eigenvalues)[::-1]
		sorted_eigenvectors = eigenvectors[:, sorted_indices]
		self.E= sorted_eigenvectors[:,-self.k:]
		# plot_cdf(eigenvalues)
		

	def compress(self,data):
		func = np.vectorize(lambda x,E: np.matmul(x,E), signature="(n),(n,m)->(m)")
		return func(data,self.E)
	def project(self,x):
		projection = np.zeros(len(x))
		es = self.E.T
		print(projection.shape)
		for e in es:
			projection += np.dot(x, e) * e
		return projection
	def decompress_image(self,data):
		# since we required <e,e>=1 this is a valid decompression
		decompress_func = np.vectorize(lambda c, E: np.dot(c, E.T), signature='(n,k),(d,k)->(n,d)')
		reconstructed_data = decompress_func(data,self.E)
		return reconstructed_data


class kNN():
	def __init__(self, train_samples,train_labels,k):
		self.k=k
		self.x= train_samples
		self.y = train_labels
		self.norms = (np.vectorize(lambda x: np.dot(x,x),signature="(n)->()")(train_samples))
		print(self.norms.shape)
	def classify(self,sample):
		dis = (np.vectorize(lambda x, y, norm: (np.dot(x, y) - norm), signature='(n),(n),()->()')(sample,self.x,self.norms))
		knn_indices= np.argpartition(dis, self.k)[:self.k]
		knn_labels = self.y[knn_indices]
		_, counts = np.unique(knn_labels, return_counts=True)
		max_count_index = np.argmax(counts)
		return counts[max_count_index]
	

		
def plot_picture(pixels):
    # convert the list of pixels to a numpy array
    pixels = np.array(pixels)
    
    # determine the dimensions of the picture
    num_pixels = len(pixels)
    dim = int(np.sqrt(num_pixels))
    
    # reshape the array into a 2D array with the correct dimensions
    pixels = pixels.reshape(dim, dim)
    
    # plot the picture using matplotlib
    plt.imshow(pixels, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
	train_samples, train_labels = load("fashion-mnist_train.csv")
	test_samples,test_labels = load("fashion-mnist_test.csv")
	t = time()
	pca=PCA(train_samples,15)
	num = 10000
	compressed_train = pca.compress(train_samples[0:num])
	compressed_test = pca.compress(test_samples[0:num])
	model=kNN(compressed_train,train_labels[0:num], 4)
	print(test(model,compressed_test,test_labels[0:num]))
	print(time()-t)
	
	

	#classify test set
	# predicted_labels=knn_model.classify_dataset(pca_model.convert(x_test))

	#test accuracy
	# correct_count=0
	# for i in range(y_test.shape):
	# 	if(y_test[i]==predicted_labels[i]):
	# 		correct_count+=1

	# accuracy=correct_count/y_test.shape

	#plt.show()

	# print(f"Test accuracy is: {accuracy * 100}%")

