import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

class PCAImplementation:
    def __init__(self, data, k):
        self.k = k
        S = (data.shape[0]) * np.cov(np.stack(data).transpose())
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        self.E = sorted_eigenvectors[:, :self.k]

    def compress(self, data):
        func = np.vectorize(lambda x, E: np.matmul(x, E), signature="(n),(n,m)->(m)")
        return func(data, self.E)

    def decompress_image(self, data):
        decompress_func = np.vectorize(lambda c, E: np.dot(c, E.T), signature='(n,k),(d,k)->(n,d)')
        reconstructed_data = decompress_func(data, self.E)
        return reconstructed_data


def load_data(path):
    data = pd.read_csv(path)
    labels = data.iloc[:, 0].values
    features = data.iloc[:, 1:].values
    return labels, features

def plot_image(ax, pixels, title,k):
    ax.imshow(pixels.reshape(k,k), cmap='gray')
    ax.axis('off')
    ax.set_title(title)

if __name__ == '__main__':
    # Load data
    labels, features = load_data("fashion-mnist_train.csv")

    # Set the desired number of components
    k = 225

    # Perform PCA using sklearn
    sklearn_pca = PCA(n_components=k)
    compressed_sklearn = sklearn_pca.fit_transform(features)
    reconstructed_sklearn = sklearn_pca.inverse_transform(compressed_sklearn)
    projection_sklearn = np.dot(compressed_sklearn, sklearn_pca.components_)

    # Perform PCA using your implementation
    pca_implementation = PCAImplementation(features, k)
    compressed_implementation = pca_implementation.compress(features)
    reconstructed_implementation = pca_implementation.decompress_image(compressed_implementation)
    projection_implementation = np.dot(compressed_implementation, pca_implementation.E.T)

    # Plot the images
    fig, axs = plt.subplots(2, 4, figsize=(6, 12))

    # Original image
    plot_image(axs[0,0], features[0], 'Original',28)

    # Sklearn PCA
    # plot_image(axs[0,1], compressed_sklearn[0], f'Sklearn Compressed (k={k})',15)
    # plot_image(axs[0,2], reconstructed_sklearn[0], 'Sklearn Decompressed',28)
    # plot_image(axs[0,3], projection_sklearn[0], 'Sklearn Projection',28)

    # Your PCA implementation
    plot_image(axs[0,1], compressed_implementation[0], f'Compressed (k={k})',15)
    plot_image(axs[0,2], reconstructed_implementation[0], 'Decompressed',28)
    plot_image(axs[0,3], projection_implementation[0], 'Projection',28)

    plt.tight_layout()
    plt.show()
