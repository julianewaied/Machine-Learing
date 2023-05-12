# #!/usr/bin/python
# # -*- coding: utf-8 -*-
# import numpy as np
# import pandas as pd
# import math
# from sklearn.feature_extraction.text import CountVectorizer


# def readTrainData(file_name):
#     file = open(file_name, 'r')
#     lines = file.readlines()
#     texAll = []
#     lbAll = []
#     voc = []
#     for line in lines:
#         splitted = line.split('\t')
#         lbAll.append(splitted[0])
#         texAll.append(splitted[1].split())
#         words = splitted[1].split()
#         for w in words:
#             voc.append(w)
#     voc = set(voc)
#     cat = set(lbAll)
#     return texAll, lbAll, voc, cat


# texAll, lbAll, voc, cat = readTrainData('r8-train-stemmed.txt')


# def learn_NB_text():

#     # calculating prrior

#     P = [lbAll.count(cur_cat) for cur_cat in cat]
#     P = np.array(P)
#     P = P / len(lbAll)

#     matrix = np.zeros((len(cat), len(voc) + 1))
#     voc_exr = list(voc.copy())
#     voc_exr.append('UNKNOWN')
#     i = 0
#     for cur_cat in cat:

#         # creating list of sentences which belong to current catagory only

#         catagory_docs = [' '.join(texAll[index]) for (index,
#                          lbl_name) in enumerate(lbAll) if lbl_name
#                          == cur_cat]

#         vec_c = CountVectorizer(vocabulary=list(voc))
#         X_c = vec_c.fit_transform(catagory_docs)  # returns 2D array which we saw on last Tutorial - how many times a word appeared in each sentence
#         words_c = X_c.toarray().sum(axis=0)  # returns 1D array which says how many copies of words there are in total in all sentences from current category

#         # using Laplace Smoothing

#         words_num_in_catagory = words_c.sum()
#         words_p = np.array(words_c, dtype=float)
#         words_p = words_c + 1  # using laplace numerator
#         words_p = words_p / (words_num_in_catagory + len(voc))  # using laplace denominator

#         # creating probabilities matrix with additoinal column of unknown words (0 occurences)

#         matrix[i] = np.append(words_p, 1 / (words_num_in_catagory
#                               + len(voc)))
#         i += 1

#     Pw = pd.DataFrame(matrix, index=cat, columns=voc_exr)  # adding headers to data

#     return Pw, P


# texAll2, lbAll2, voc2, cat2 = readTrainData('r8-test-stemmed.txt')


# def ClassifyNB_text(Pw, P):

#     # all sentences in test

#     sum_right = 0
#     for index, sentence in enumerate(texAll2):
#         max_cat_sum = -math.inf
#         max_cat_name = ''

#         # all catagories - for comparing and deciding

#         for ic, catagory in enumerate(cat):

#             # calculating array with all the probabiliets of P(word|catagory)

#             sentence_prob_array = np.array([(Pw[word][catagory] if word
#                     in voc else Pw['UNKNOWN'][catagory]) for word in
#                     sentence])

#             # instead of doing P(cat)*mul(P(word1|catagory),...)
#             # we used log! saving calculating time

#             cat_sum_new = np.log(sentence_prob_array).sum()  # log the results and sum them
#             currect_prob = cat_sum_new + math.log(P[ic])  # bayesian theorem

#             # getting max cat sum and it's name

#             if currect_prob > max_cat_sum:
#                 max_cat_sum = currect_prob
#                 max_cat_name = catagory

#         # count number of correctly classified sentences

#         if max_cat_name == lbAll2[index]:
#             sum_right += 1

#     sum_right = sum_right / len(lbAll2)
#     return sum_right


# Pw, P = learn_NB_text()
# sum_right = ClassifyNB_text(Pw, P)
# print(sum_right)

