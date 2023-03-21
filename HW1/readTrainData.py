import numpy as np
import seaborn as sbs
labels = []
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
def find_class_conditional(word, label, voc):
    # returns an array of class conditional (Pw), should use laplace smoothing
    return 0
def find_prior(label,voc):
    # returns the prior of a label (P)
    return 0
def classify(sentence,Pw,P):
    # use MAP and find the most probable class 
    return 0
def classify_NB_test(Pw,P):
    return 0

print('hello world')


