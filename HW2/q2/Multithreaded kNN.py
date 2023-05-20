import numpy as np
import pandas as pd
import time
import threading
import queue
import matplotlib.pyplot as plt
q = queue.Queue()
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def plot_function(x_values, y_values):
    # create a figure and axis objects
    fig, ax = plt.subplots()

    # plot the function
    ax.plot(x_values, y_values)

    # set axis labels and title
    ax.set_xlabel('k')
    ax.set_ylabel('accuracy')
    ax.set_title('Finding K')

    # show the plot
    plt.show()

def test_model(model,path):
    all_data = pd.read_csv(path)
    all_data = all_data.dropna().tail(500)
    data = all_data.to_numpy()
    samples  = data[:,1:]
    labels = data[:,0] 
    succ = 0
    for i in range(0,data.shape[0]):
        pred = model.classify(samples[i])
        if(pred == labels[i]):
            succ +=1
        else:
            print('label is:' ,labels[i], 'prediction is', pred)
            plot_picture(samples[i].tolist())
    print(samples.shape[0])
    return succ/samples.shape[0]

def test(model,data,label):
    ans = model.classify(data)
    if( ans == label):
        q.put(1)
    else:
        # print(ans)
        # plot_picture(data)
        q.put(0)
def plot_picture(pixels):
    # convert the list of pixels to a numpy array
    pixels = np.array(pixels)
    
    # determine the dimensions of the picture
    num_pixels = len(pixels)
    dim = int(np.sqrt(num_pixels))
    
    # reshape the array into a 2D array with the correct dimensions
    pixels = pixels.reshape(dim, dim)
    
    # plot the picture using matplotlib
    plt.imshow(pixels, cmap='gray')
    plt.axis('off')
    plt.show()
class KNNClassifier:
    def __init__(self, k):
        self.yair = k # which is K
        self.oren = [] # training data
        self.julian = []
    def train(self,path):
         tv_data = pd.read_csv(path)
         tv_data = tv_data.dropna()
         data_train, data_valid = train_test_split(tv_data, test_size=0.01, random_state=40, shuffle=True)
         self.oren = data_train.to_numpy()[0:500,:]
        #  self.oren = tv_data.to_numpy()
         self.julian = data_valid.to_numpy()
         self.labels = set(self.oren[:,0])
        #  print(self.oren)
    def classify(self,sample):
        distances = self.oren[:,1:]
        distances = [distances[i]-sample for i in range(0,len(distances))]
        distances = np.linalg.norm( distances,axis=1)
        all_data = [[self.oren[i][0],distances[i]] for i in range(0,len(distances)) ]
        all_data = np.array(all_data)
        all_data = (all_data[all_data[:,1].argsort()])[0:self.yair,0]
        p = 0
        answer = 'I dunno'
        for label in self.labels:
            c = list(all_data).count(label)
            if(c>p):
                answer = label
                p = c
        return answer 
    def validate(self):
        test_data = self.julian[:,1:]
        test_labs = self.julian[:,0]
        correct = 0
        threads = []
        for i in range(0,len(test_data)):
            thread = threading.Thread(target=test,args=(model,test_data[i],test_labs[i]))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
            correct += q.get()

        return correct/len(test_data)
    def setK(self,k):
        self.yair = k
train_path = "train.csv"
test_path = "test.csv"

st = time.time()

model = KNNClassifier(4)
model.train(train_path)
print('done training!')
print('testing...')
print('accuracy is : ',test_model(model,train_path))

print(time.time()-st)


