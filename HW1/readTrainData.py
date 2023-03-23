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
    for label in cat:
        class_conditionals[label]={}
        for word in voc:
            class_conditionals[label][word] = 1
    # counting number of class conditionals
    for i in range(0,len(texAll)):
        for word in texAll[i]:
            class_conditionals[lbAll[i]][word] = class_conditionals[lbAll[i]][word] + 1
    # applying laplace smoothing
    for label in cat:
        count = lbAll.count(label) + 2
        for word in voc:
            class_conditionals[label][word] = log2((class_conditionals[label][word] + 1) / count)
    return class_conditionals
def find_prior(cat,lbAll):
    # returns the prior of a label (P)
    # counts has the likelihood for new words
    sum = 0
    priors = {}
    counts = {}
    for label in cat:
        count = lbAll.count(label)
        prior = count/len(lbAll)
        priors[label] = log2(prior)
        counts[label] = -log2((count +2))
    return priors, counts
def calc_posterior(sentence,label,Pw,P,counts,voc):
    p = P[label]
    for word in sentence:
        # summing log2 of likelihood times prior
        if word in voc:
            p = p + Pw[label][word]
        else:
            p = p + counts[label]
    print(sentence,label,2**p)
    return p
def classify(sentence,Pw,P,cat,counts,voc):
    # use MAP and find the most probable class 
    max_p  = -1.0E40
    max_label = 'dunno'
    for label in cat:
        p = calc_posterior(sentence,label,Pw,P,counts,voc)
        if(p>max_p):
            max_p = p
            max_label = label
    return max_label
def classify_NB_test(Pw,P,cat,counts):
    return 0

print('hello world')
texAll, lbAll, voc, cat = readTrainData("train-test.txt")
Pw = find_class_conditional(texAll,lbAll,voc,cat)
P,counts = find_prior(cat,lbAll)
error = 0
texAll1, lbAll1, voc1, cat1 = readTrainData("test-test.txt")
for i in range(0,len(texAll1)):
    label = classify(texAll1[i],Pw,P,cat,counts,voc)
    if(label != lbAll1[i]):
        error+=1
print('error rate is : ', error/len(texAll1))
