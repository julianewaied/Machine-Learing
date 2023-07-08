import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

# Copy here your full decision tree from q1
class DecisionTree:
	def __init__(self):
		self.tree = {}

	# Calculate the entropy of a given dataset, the distribution is over the target class.
	def calculate_entropy(self, data):
		labels = data.iloc[:, -1]
		# Your code goes here

	# Calculate the information gain of a feature based on its value
	def calculate_information_gain(self, data, feature):
		total_entropy = self.calculate_entropy(data)
		information_gain = total_entropy
		
		distincts = list(set(data[feature]))		#get the values of the feature
		
		# Your code goes here
		
		return information_gain

	def filter_data(self, data, feature, value):
		return data[data[feature] == value].drop(feature, axis=1)

	def create_tree(self, data, depth=0):
		# Recursive function to create the decision tree
		labels = data.iloc[:, -1]

		# Base case: if all labels are the same, return the label
		if len(np.unique(labels)) == 1:
			return list(labels)[0]

		features = data.columns.tolist()[:-1]
		
		# Base case: if there are no features left to split on, return the majority label
		if len(features) == 0:
			unique_labels, label_counts = np.unique(labels, return_counts=True)
			majority_label = unique_labels[label_counts.argmax()]
			return majority_label
		
		selected_feature = None
		best_gain = 0
		
		# Select feature that maximizes gain
		for feature in features:
			gain = self.calculate_information_gain(data, feature)
			if gain > best_gain:
				selected_feature = feature
				best_gain = gain

		# Create the tree node
		tree_node = {}
		
		distincts = list(set(data[selected_feature]))
		for value in distincts:
			new_data = self.filter_data(data, selected_feature, value)
			tree_node[(selected_feature, value)] = self.create_tree(new_data)

		return tree_node

	def fit(self, data):
		self.tree = self.create_tree(data)

	def predict(self, X):
		X = [row[1] for row in X.iterrows()]
	
		# Predict the labels for new data points
		predictions = []

		for row in X:
			current_node = self.tree
			while isinstance(current_node, dict):
				split_condition = next(iter(current_node))
				feature, value = split_condition
				current_node = current_node[feature, row[feature]]
			predictions.append(current_node)

		return predictions
	
	def _plot(self, tree, indent):
		depth = 1
		for key, value in tree.items():
			if isinstance(value, dict):
				print(" " * indent + str(key) + ":")
				depth = max(depth, 1 + self._plot(value, indent + 2))
			else:
				print(" " * indent + str(key) + ": " + str(value))
		return depth
	
	def plot(self):
		depth = self._plot(self.tree, 0)
		print(f'depth is {depth}')


class RandomForest:
	def __init__(self, maxdepth=np.inf, n_estimators=3, method='simple'):
		self.forest = []
		self.maxdepth = maxdepth
		self.n_estimators = n_estimators
		self.method = method

	def select_features(self, data):
		np.random.seed(40+len(self.forest))
		
		if self.method == 'sqrt':
			m = int(np.sqrt(len(data.columns)-1))
		elif self.method == 'log':
			m = int(np.log2(len(data.columns)-1))
		else:
			m = np.random.randint(0, len(data.columns))

		incidies = np.random.choice(np.arange(0, len(data.columns)-1), size=m, replace=False)
		features = list(data.columns[incidies])
		return data[features + ['class']]

	def fit(self, data):
		self.forest = []
		# Your code goes here
		self.forest= [DecisionTree() for i in range(self.n_estimators)]
		
		for tree in self.forest:
			tree.fit(self.selected_features(data))


	def _predict(self, X):
		# Predict the labels for new data points
		predictions = []

		preds = [tree.predict(X) for tree in self.forest]
		preds = list(zip(*preds))
		predictions = [Counter(est).most_common(1)[0][0] for est in preds]

		return predictions
		
	def score(self, X, sample_weight=None):
		pred = self._predict(X)
		return (pred == X.iloc[:,-1]).sum() / len(X)


def KFold(data, model, cv=5):
	chunks = np.array_split(data, cv)
	correct_sum = 0
	# Your code goes here
	for i in range(cv):
		test_data=chunks[i]
		training_data=get_training_data(chunks,i,cv)

		correct_sum+= count_correct(training_data,test_data,model)


	return correct_sum / cv

def count_correct(training_data,test_data,model):
	model.fit(training_data)
	predicted_labels=model._predict(test_data)
	return np.sum(predicted_labels == test_data[,1:1])

def get_training_data(chunks,i,cv):
	
	training_data=np.empty()
	for j in range(cv):
		if i!=j:
			training_data.append(chunks[i])

	return training_data

data = pd.read_csv('cars.csv')

train, test = train_test_split(data, test_size=0.2, random_state=13)

'''					SECTION B 					'''
# Run single decision tree:
# tree = DecisionTree(maxdepth=3)
# tree.fit(train)

# pred = tree.predict(train)
# acc = (pred == train.iloc[:,-1]).sum() / len(train)
# print(f'Decision Tree Training accuracy is {acc}')

# pred = tree.predict(test)
# acc = (pred == test.iloc[:,-1]).sum() / len(test)
# print(f'Decision Tree Test accuracy is {acc}')
# print()

# Run random forest
# forest = RandomForest(maxdepth=3, n_estimators=5, method='simple')
# forest.fit(train)

# acc = forest.score(train)
# print(f'Random Forest Training accuracy is {acc}')

# acc = forest.score(test)
# print(f'Random Forest Test accuracy is {acc}')

'''					SECTION C 					'''

# correct = []

# for i in range(1,8):
	# forest = RandomForest(maxdepth=3, n_estimators=i, method='simple')
	# correct.append(KFold(data=train, model=forest, cv=5))

# plt.plot(range(1,8), np.array(correct))
# plt.xlabel('trees num')
# plt.ylabel('avg accuracy')
# plt.show()

