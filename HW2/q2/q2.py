import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

class PCA()

accuracy = 0
print(f"Test accuracy is: {accuracy * 100}%")
