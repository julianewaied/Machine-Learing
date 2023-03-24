import math
import time
# define function to read data from file
def read_data(file_path):
    labels = []
    documents = []
    with open(file_path, 'r') as file:
        for line in file:
            label, document = line.strip().split('\t')
            labels.append(label)
            documents.append(document)
    return labels, documents

# define function to preprocess text data
def preprocess_data(documents):
    # split documents into words
    words_list = [doc.split() for doc in documents]
    # flatten the list of words
    words = [word for words in words_list for word in words]
    # create vocabulary (set of unique words)
    vocabulary = set(words)
    return words_list, vocabulary

# define Naive Bayes classifier
class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.posteriors = {}
        self.classes = set()
        self.vocabulary = set()
        
    # train the model
    def train(self, labels, documents, alpha=1):
        # preprocess data
        words_list, self.vocabulary = preprocess_data(documents)
        # calculate priors
        N = len(labels)
        for label in labels:
            if label not in self.priors:
                self.priors[label] = 0
            self.priors[label] += 1
            self.classes.add(label)
        for label in self.classes:
            self.priors[label] = (self.priors[label] + alpha) / (N + alpha*len(self.classes))
        # calculate posteriors
        word_counts = {}
        for label in self.classes:
            self.posteriors[label] = {}
            for word in self.vocabulary:
                self.posteriors[label][word] = 0
                word_counts[word] = 0
        for i in range(N):
            label = labels[i]
            for word in words_list[i]:
                self.posteriors[label][word] += 1
                word_counts[word] += 1
        for label in self.classes:
            for word in self.vocabulary:
                self.posteriors[label][word] = (self.posteriors[label][word] + alpha) / (word_counts[word] + alpha*len(self.vocabulary))
            
    # predict the label of a document
    def predict(self, document):
        words = document.split()
        scores = {}
        for label in self.classes:
            scores[label] = math.log(self.priors[label])
            for word in words:
                if word in self.posteriors[label]:
                    scores[label] += math.log(self.posteriors[label][word])
        return max(scores, key=scores.get)
    
    # test the model on a set of documents
    def test(self, labels, documents):
        correct = 0
        total = len(labels)
        for i in range(total):
            predicted_label = self.predict(documents[i])
            if predicted_label == labels[i]:
                correct += 1
        accuracy = correct / total
        return accuracy
start_time = time.time()
# read training data
train_labels, train_documents = read_data("C:/Users/WIN10PRO/Desktop/My Stuff/University/BSC/Machine Learning/Machine-Learing/HW1/r8-train-stemmed.txt")

# train the classifier
nb = NaiveBayes()
nb.train(train_labels, train_documents)

# read testing data
test_labels, test_documents = read_data("C:/Users/WIN10PRO/Desktop/My Stuff/University/BSC/Machine Learning/Machine-Learing/HW1/r8-test-stemmed.txt")

# test the classifier
accuracy = nb.test(test_labels, test_documents)

# print the accuracy rate
print('Accuracy:', accuracy)
print('execution time : ',time.time()-start_time)