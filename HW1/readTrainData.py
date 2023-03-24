import numpy as np
from math import log2
import time
def readTrainData(file_name):
    file = open(file_name, 'r')
    lines = file.readlines()
    texAll = []
    lbAll = []
    voc = []
    for line in lines:
        splitted = line.split('\t')
        # this is all labels
        lbAll.append(splitted[0])
        # this is all words 
        texAll.append(splitted[1].split())
        words = splitted[1].split()
        for w in words:
            voc.append(w)
    voc = set(voc)
    cat = set(lbAll)
    return texAll, lbAll, voc, cat
def classify_NB_test(model,texAll,lbAll,voc,cat):
    success = 0
    for i in range(0,len(texAll)):
        label = model.classify(texAll[i])
        if(label == lbAll[i]):
           success+=1
    return success/len(texAll)
class NBClassifier:
    def __init__(self,texAll,lbAll,voc,cat):
        self.texAll = texAll
        self.lbAll = lbAll
        self.voc = voc
        self.cat = cat
        self.counts = {}
        for label in cat:
            self.counts[label] = 0
        for i in range(0,len(lbAll)):
            self.counts[lbAll[i]]+=1
        self.totals = {}
        for label in cat:
            self.totals[label] = 0
        for i in range(0,len(lbAll)):
            self.totals[lbAll[i]]+=len(texAll[i])
    def find_class_conditional(self):
        # returns a nested dictionary of class conditional (Pw), should use laplace smoothing
        self.Pw = {}
        # creating nested dict
        for label in self.cat:
           self.Pw[label]={}
           for word in self.voc:
              self.Pw[label][word] = 1
        # counting number of class conditionals
        for i in range(0,len(self.texAll)):
            for word in self.texAll[i]:
                self.Pw[self.lbAll[i]][word] = self.Pw[self.lbAll[i]][word] + 1
        # applying laplace smoothing
        for label in self.cat:
            count = self.totals[label] + 2
            for word in self.voc:
                self.Pw[label][word] = np.log2(self.Pw[label][word] / count)
    def find_prior(self):
        # returns the prior of a label (P)
        # self.neutrals has the likelihood for new words
        sum = 0
        self.P = {}
        self.neutrals = {}
        for label in self.cat:
            total = self.totals[label]
            count = self.counts[label]
            prior = count/len(lbAll)
            self.P[label] = np.log2(prior)
            self.neutrals[label] = np.log2(1/(total +2))
    def calc_posterior(self,sentence,label):
        p = self.P[label]
        for word in sentence:
            # summing log2 of likelihood times prior
            if (word in self.voc) and (self.Pw[label][word] != 0):
                p = p + self.Pw[label][word]
            else:
                p = p + self.neutrals[label]
        return p
    def classify(self,sentence):
        # use MAP and find the most probable class 
        max_p  = -1.0E40
        max_label = 'dunno'
        for label in self.cat:
            p = self.calc_posterior(sentence,label)
            if(p>max_p):
                max_p = p
                max_label = label
        return max_label
    def train(self):
        self.find_class_conditional()
        self.find_prior()
print('begin')
st = time.time()
texAll, lbAll, voc, cat = readTrainData("r8-train-stemmed.txt")
model = NBClassifier(texAll,lbAll,voc,cat)
model.train()
texAll1, lbAll1, voc1, cat1 = readTrainData("r8-test-stemmed.txt")
acc_rate = classify_NB_test(model,texAll1, lbAll1, voc1, cat1)
print('success rate is  : ',acc_rate )
print('execution time : ',time.time()-st)
