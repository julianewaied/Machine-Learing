import numpy as np
import seaborn as sbs
from math import log2
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
def find_class_conditional(texAll,lbAll,voc,cat):
    # returns a nested dictionary of class conditional (Pw), should use laplace smoothing
    # class conditionals: P(word | label)
    # we will return the log of it
    class_conditionals = {}
    # creating nested dict
    for word in voc:
        for label in cat:
            if label not in class_conditionals:
                class_conditionals[label]={}
            class_conditionals[label][word] = 1
    # counting number of class conditionals
    for i in range(0,len(texAll)):
        for word in texAll[i]:
            class_conditionals[lbAll[i]][word] += 1
    # applying laplace smoothing
    for label in cat:
        count = lbAll.count(label) + 2
        for word in voc:
            class_conditionals[label][word] = log2((class_conditionals[label][word] + 1) / count)
    return class_conditionals            

    return 0
def find_prior(cat,lbAll):
    # returns the prior of a label (P)
    sum = 0
    priors = {}
    for label in cat:
        prior = lbAll.count(label)/len(lbAll)
        priors[label] = log2(prior)
    return priors
    return 0
def calc_posterior(sentence,Pw,P,label,lbAll):
    words = sentence.split()
    sum = 0
    
    return 0
def classify(sentence,Pw,P):
    # use MAP and find the most probable class 
    return 0
def classify_NB_test(Pw,P):
    return 0

print('hello world')
texAll, lbAll, voc, cat = readTrainData("r8-train-stemmed.txt")
print(find_class_conditional(texAll,lbAll,voc,cat)[cat.pop()][voc.pop()])
