import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from time import sleep
import time
from q3 import logistic_regression, load, preprocess,evaluate
# VS code runs idk, cmd runs q3
# Load data and preprocess it
data_path = "C:/Users/WIN10PRO/Desktop/My Stuff/University/BSC/Machine Learning/Machine-Learing/HW2/q3/exams.csv"

# Define the range of learning rates to test
learning_rates = np.linspace(0.5, 10, num=96)
t = time.time()
# Test each learning rate and save the accuracy rates
accuracy_rates = []
for lr in learning_rates:
    data = load(data_path)
    data = data.head(90)
    # 7 -> 80%
    model = logistic_regression(convergance_constant=0.357,learning_rate=lr)
    accuracy, w = evaluate(model,data)
    accuracy_rates.append(accuracy)
# Find the best learning rate
best_lr = learning_rates[np.argmax(accuracy_rates)]
print(f"Best learning rate: {best_lr}")

# Plot the accuracy rates as a function of the learning rate
plt.plot(learning_rates, accuracy_rates)
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Learning Rate')
print(f'execution time is : {time.time()-t}')
plt.show()