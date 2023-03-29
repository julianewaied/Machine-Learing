import math

class NaiveBayesClassifier:
    def __init__(self):
        self.class_counts = {} # counts of each class
        self.word_counts = {} # counts of each word in each class
        self.vocab = set() # unique words in the training set

    def train(self, data):
        for line in data:
            label, sentence = line.strip().split('\t')
            if label not in self.class_counts:
                self.class_counts[label] = 0
                self.word_counts[label] = {}
            self.class_counts[label] += 1
            for word in sentence.split():
                if word not in self.vocab:
                    self.vocab.add(word)
                if word not in self.word_counts[label]:
                    self.word_counts[label][word] = 0
                self.word_counts[label][word] += 1

    def classify(self, sentence):
        scores = {}
        for label in self.class_counts:
            score = math.log(self.class_counts[label])
            for word in sentence.split():
                if word in self.word_counts[label]:
                    count = self.word_counts[label][word]
                    score += math.log((count) / (self.class_counts[label]))
                else:
                    count = 0
                    score += math.log((count + 1) / (self.class_counts[label] ))
            scores[label] = score
        return max(scores, key=scores.get)

    def evaluate(self, data):
        correct = 0
        total = 0
        for line in data:
            label, sentence = line.strip().split('\t')
            predicted_label = self.classify(sentence)
            if predicted_label == label:
                correct += 1
            total += 1
        accuracy = correct / total
        return accuracy
# read in the training data
with open("r8-train-stemmed.txt", 'r') as f:
    train_data = f.readlines()

# read in the test data
with open("r8-test-stemmed.txt", 'r') as f:
    test_data = f.readlines()

# train the classifier
classifier = NaiveBayesClassifier()
classifier.train(train_data)

# evaluate the classifier on the test set
accuracy = classifier.evaluate(test_data)
print(f'Accuracy: {accuracy}')
