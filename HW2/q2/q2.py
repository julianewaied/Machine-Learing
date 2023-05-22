import numpy as np
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


class PCA():

	def __init__(self,data,original_size,new_size):
		self.new_size=new_size
		self.original_size=original_size
		self.d=original_size**2
		self.k=new_size**2

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
		decompress_func = np.vectorize(lambda c, E: np.dot(c, E.T), signature='(n,k),(d,k)->(n,d)')
		reconstructed_data = decompress_func(data,self.E)
		return reconstructed_data


class KNN():
	def __init__(self,x_train,y_train,k):
		self.k=k
		self.x_train=x_train.reshape((x_train.shape[0]),-1)
		self.y_train=y_train
		self.distance_label_pairs= np.empty((self.x_train.shape[0],2))
		
	def classify_dataset(self,data):
		result_labels=np.empty(data.shape[0])
		for i in range(result_labels.shape[0]):
			result_labels[i]= self.classify_sample(data[i])

		return result_labels


	def classify_sample(self,test_sample):
		neighbours_labels_counter= np.zeros(10)
		
		for i in range(self.x_train.shape[0]):
			self.distance_label_pairs[i][0]=self.euclidian_distance(test_sample.reshape(-1),self.x_train[i])
			self.distance_label_pairs[i][1]=self.y_train[i]

		#find k nearest neighbours
		print("here1")	
		np.sort(self.distance_label_pairs)
		print("here2")
		for i in range(self.k):
			neighbours_labels_counter[int(self.distance_label_pairs[i][1])]+=1
		#print("done")

		return neighbours_labels_counter.argmax()


	def euclidian_distance(self,sample1,sample2):
		return 	np.linalg.norm(sample1-sample2,axis=0)

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
	t = time()
	pca=PCA(train_samples,28,15)
	compressed_samples = pca.compress(train_samples)
	print(time()-t)





	# pca_x_train=pca_model.convert(x_train)

	#f, axarr = plt.subplots(2,1) 	
	#axarr[0].imshow(model.decompress_image(reduced_train[2]),cmap="gray")
	#axarr[1].imshow(x_train[2],cmap="gray")
	
	# knn_model=KNN(pca_x_train,y_train,1)  #set K

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

