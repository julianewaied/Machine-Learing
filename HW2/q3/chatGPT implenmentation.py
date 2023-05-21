import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time

# Import functions from test.py
from test import load, preprocess, logistic_regression, evaluate

data_path = "exams.csv"

# Load and preprocess the data
data = load(data_path)
points, labels = preprocess(data)

# Define the range of initial w coordinates and learning rates
w_range = np.linspace(2, 2, 21)
learning_rate_range = np.linspace(1, 7, 50)

# Create a grid of initial w coordinates and learning rates
W1, W2, W3 = np.meshgrid(w_range, w_range, w_range)
learning_rates = np.meshgrid(learning_rate_range)[0]

# Flatten the coordinate arrays and learning rates for iteration
W1_flat = W1.flatten()
W2_flat = W2.flatten()
W3_flat = W3.flatten()
learning_rates_flat = learning_rates.flatten()

accuracies = []  # Store the accuracies

best_accuracy = 0
best_w = None
best_learning_rate = None

# Iterate over all combinations of initial w and learning rate
for i in range(len(W1_flat)):
    # Extract current w and learning rate
    w1 = W1_flat[i]
    w2 = W2_flat[i]
    w3 = W3_flat[i]
    learning_rate = learning_rates_flat[i]

    # Create a logistic regression model with the current parameters
    model = logistic_regression(w = [w1, w2, w3], learning_rate=learning_rate)

    # Measure the running time and accuracy
    start_time = time()
    accuracy, current_w = evaluate(model, data)
    running_time = time() - start_time

    # Update the best parameters if necessary
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_w = current_w
        best_learning_rate = learning_rate

    # Print the current combination's results
    print(f"Initial w: [{w1}, {w2}, {w3}], Learning Rate: {learning_rate}")
    print(f"Accuracy: {accuracy}, Running Time: {running_time}\n")

    # Store the accuracy
    accuracies.append(accuracy)

# Plot the accuracy as a function of the learning rate
plt.plot(learning_rates_flat, accuracies)
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.title("Accuracy as a Function of Learning Rate")
plt.show()

# Print the best w and learning rate
print("Best Parameters:")
print(f"Initial w: {best_w}")
print(f"Learning Rate: {best_learning_rate}")
