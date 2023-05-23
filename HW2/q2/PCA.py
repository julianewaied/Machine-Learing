import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Load the dataset
train_data = np.loadtxt("fashion-mnist_train.csv", delimiter=",", skiprows=1)
test_data = np.loadtxt("fashion-mnist_test.csv", delimiter=",", skiprows=1)

# Extract labels from the dataset
train_labels = train_data[:, 0]
test_labels = test_data[:, 0]

# Extract features from the dataset
train_features = train_data[:, 1:]
test_features = test_data[:, 1:]

# Apply PCA for dimensionality reduction
pca = PCA(n_components=15)
train_features_pca = pca.fit_transform(train_features)
test_features_pca = pca.transform(test_features)

# Create a k-NN classifier with k=8
knn = KNeighborsClassifier(n_neighbors=8)

# Train the classifier
knn.fit(train_features_pca, train_labels)

# Make predictions on the test set
predictions = knn.predict(test_features_pca)

# Calculate the accuracy of the predictions
accuracy = accuracy_score(test_labels, predictions)

# Print the accuracy
print(f"Test accuracy is: {accuracy * 100}%")
