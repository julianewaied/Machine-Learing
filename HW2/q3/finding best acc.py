import numpy as np
import matplotlib.pyplot as plt
from q3 import logistic_regression, load, preprocess, evaluate

def find_best_learning_rate(data, learning_rates):
    accuracies = []
    for learning_rate in learning_rates:
        model = logistic_regression(learning_rate=learning_rate)
        accuracy, _ = evaluate(model, data)
        accuracies.append(accuracy)
    best_idx = np.argmax(accuracies)
    best_learning_rate = learning_rates[best_idx]
    best_accuracy = accuracies[best_idx]
    return best_learning_rate, best_accuracy, accuracies

# Load data
data_path = "exams.csv"
data = load(data_path)

# Define learning rates to test as a linspace
num_learning_rates = 20
min_learning_rate = 0.1
max_learning_rate = 10
learning_rates = np.linspace(min_learning_rate, max_learning_rate, num_learning_rates).append(6.35263157894736)

# Find the best learning rate
best_learning_rate, best_accuracy, accuracies = find_best_learning_rate(data, learning_rates)

# Plot accuracy as a function of learning rate
plt.plot(learning_rates, accuracies, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy as a Function of Learning Rate')
plt.grid(True)
plt.show()

print("Best Learning Rate:", best_learning_rate)
print("Best Accuracy:", best_accuracy)
