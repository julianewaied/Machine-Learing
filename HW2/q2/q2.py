import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import tensorflow as tf
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


class PCA():

	def __init__(self,data,original_size,new_size):
		self.new_size=new_size
		self.original_size=original_size
		self.d=original_size*original_size
		self.k=new_size*new_size

		#flatten features
		flattened_data= data.reshape((data.shape[0],-1))

		#normalize data
		mean= flattened_data.mean(axis=0)
		Z= flattened_data-mean


		#compute scatter matrix, eigenvalues, eigenvectors

		S= np.matmul(Z.transpose(),Z)
		eigenvalues,eigenvectors = np.linalg.eigh(S)

		#plot_cdf(eigenvalues)
		#get the k eigenvectors with the largest eigenvalues
		
		self.E= eigenvectors[:,-self.k:].transpose()	

	def convert(self,data):
		#reduce data to smaller dimention
		feature_vectors= np.matmul(self.E, data.reshape((data.shape[0],-1)).transpose()).transpose()
		return feature_vectors.reshape((feature_vectors.shape[0],self.new_size,self.new_size))

	def decompress_image(self,compressed_img):
		flattened= compressed_img.reshape((self.k,-1))
		decompressed_image= np.matmul(flattened.transpose(),self.E).transpose()
		return decompressed_image.reshape(self.original_size,self.original_size)


class KNN():
	def __init__(self,x_train,y_train,k):
		self.k=k
		self.x_train=x_train.reshape((x_train.shape[0]),-1)
		self.y_train=y_train
		self.distance_label_pairs= np.empty((self.x_train.shape[0],2))
	

	@property
	def dataset_classifier(self):
		return np.vectorize(self.classify_sample,signature='(n,n)->()')

	def classify_sample(self,test_sample):
		self.neighbours_labels_counter= np.zeros(10)
		
		
		#self.distance_label_pairs[:,0]=np.vectorize(self.euclidian_distance,signature='(n),(n)->()')(self.x_train,test_sample.flatten())
		
		self.distance_label_pairs[:,1]=self.y_train
		# for i in range(self.x_train.shape[0]):
		# 	self.distance_label_pairs[i][0]=self.euclidian_distance(test_sample.reshape(-1),self.x_train[i])
		# 	self.distance_label_pairs[i][1]=self.y_train[i]

		#find k nearest neighbours
		
		np.sort(self.distance_label_pairs)
		np.vectorize(self.count_neighbour,signature='(2)->()')(self.distance_label_pairs[:self.k])

		
		# for i in range(self.k):
		# 	self.neighbours_labels_counter[int(self.distance_label_pairs[i][1])]+=1
		#print("done1\ndone2")

		return self.neighbours_labels_counter.argmax()

	def count_neighbour(self,pair):
		self.neighbours_labels_counter[int(pair[1])]+=1
		return 0

	def euclidian_distance(self,sample1,sample2):
		return 3
		#return 	np.linalg.norm(sample1-sample2,axis=0)


def test_classify_speed(classifier,sample):
	pass

if __name__ == '__main__':
	arr= np.array([1,2,3])
	arr[:1][0]=3
	print(arr)
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
	#new_x_train= np.array([x_train[i] for i in range(x_train.shape[0]) if y_train[i]==9])

	pca_model=PCA(x_train,28,5)
	pca_x_train=pca_model.convert(x_train)

	#f, axarr = plt.subplots(2,1) 	
	#axarr[0].imshow(model.decompress_image(reduced_train[2]),cmap="gray")
	#axarr[1].imshow(x_train[2],cmap="gray")
	
	knn_model=KNN(pca_x_train,y_train,1)  #set K

	#classify test set
	#predicted_labels=knn_model.dataset_classifier(pca_model.convert(x_test))

	#test accuracy
	# correct_count=0
	# for i in range(y_test.shape):
	# 	if(y_test[i]==predicted_labels[i]):
	# 		correct_count+=1

	# accuracy=correct_count/y_test.shape

	#plt.show()
	accuracy=0
	print(f"Test accuracy is: {accuracy * 100}%")



