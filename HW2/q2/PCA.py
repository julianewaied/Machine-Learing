import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from q2 import PCA, kNN, load, test

train_samples, train_labels = load("fashion-mnist_train.csv")
test_samples, test_labels = load("fashion-mnist_test.csv")
pca = PCA(train_samples, 15)
num_train = 4000
num_test = 2000
compressed_train = pca.compress(train_samples[0:num_train])
compressed_test = pca.compress(test_samples[0:num_test])

accuracy_scores = []
k_values = range(1,11)  # Try values from 0 to 15 for k

for k in k_values:
    model = kNN(compressed_train, train_labels[0:num_train], k)
    accuracy = test(model, compressed_test, test_labels[0:num_test])
    print(f'For k = {k}, acc = {accuracy}')
    accuracy_scores.append(accuracy)

plt.plot(k_values, accuracy_scores)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. k for kNN Classifier')
plt.show()
