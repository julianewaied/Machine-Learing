import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Read the CSV file into a pandas dataframe
df = pd.read_csv('C:/Users/WIN10PRO/Desktop/My Stuff/University/BSC/Machine Learning/Machine-Learing/HW2/q3/exams.csv', header=None)

# Split the dataframe into training and testing sets
train = df.iloc[:90, :]
test = df.iloc[90:, :]

# Split the training and testing sets into features (X) and labels (y)
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

# Create a logistic regression object and fit it to the training data
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Print the coefficients of the logistic regression
w = lr.coef_[0]
print('Vector w:', w)

# Calculate the accuracy of the model on the training and testing data
y_train_pred = lr.predict(X_train)
accuracy_train = np.mean(y_train_pred == y_train)
print('Training accuracy:', accuracy_train)

y_test_pred = lr.predict(X_test)
accuracy_test = np.mean(y_test_pred == y_test)
print('Testing accuracy:', accuracy_test)

# Plot the points and the LDF
plt.scatter(train.iloc[:, 0], train.iloc[:, 1], c=train.iloc[:, -1], cmap='RdBu')
plt.scatter(test.iloc[:, 0], test.iloc[:, 1], c=test.iloc[:, -1], cmap='RdBu', marker='x')
x_vals = np.array(plt.gca().get_xlim())
y_vals = -(w[0]*x_vals + lr.intercept_)/w[1]
plt.plot(x_vals, y_vals, '--')
plt.show()
