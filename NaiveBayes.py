########################################################################################################################
# First Attempt of Naive Bayes on Amzn
#    85 % accuracy on ngrams = 1. top 10k words
#    85 % accuracy on ngrams = 1. top 20k words
#    83 % accuracy on ngrams = 2. top 20k words
#    85 % accuracy on ngrams = 2. top 40k words
#
# Steps 0: Header
#

########################################################################################################################

##### Step 0: Header #####

import re
import pandas as pd
import numpy as np
import os, re
import bz2
#import gc

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

test_file = bz2.BZ2File('C:\\Users\szreb\Documents\CodeSandBox\Amazon\data\Input\\test.ft.txt.bz2')
train_file = bz2.BZ2File('C:\\Users\szreb\Documents\CodeSandBox\Amazon\data\Input\\train.ft.txt.bz2')

train_file_lines = train_file.readlines()
test_file_lines = test_file.readlines()

del train_file, test_file
gc.collect()

##### Step 1: read in file ######

train_file_lines = [x.decode('utf-8') for x in train_file_lines]
test_file_lines = [x.decode('utf-8') for x in test_file_lines]

train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file_lines]
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file_lines]

for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d', '0', train_sentences[i])

test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file_lines]
test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file_lines]

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d', '0', test_sentences[i])

for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

for i in range(len(test_sentences)):
    if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in test_sentences[i]:
        test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

del train_file_lines, test_file_lines
gc.collect()

##### more cleaning would be better #####

##### tokenize data #####

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score


Vectorizer = CountVectorizer(lowercase = True,
                             stop_words = 'english',
                             max_features = 40000,
                             ngram_range=(2, 2))
TrainX = Vectorizer.fit_transform(train_sentences)
TestX = Vectorizer.transform(test_sentences)

AmznNaiveBayes = BernoulliNB()
AmznNaiveBayes = AmznNaiveBayes.fit(TrainX, train_labels)

TrainPred = AmznNaiveBayes.predict_proba(TrainX)
TestPred = AmznNaiveBayes.predict_proba(TestX)

TrainBinary = np.asarray(train_labels)
TrainPred2 = TrainPred[:,1]
TrainPred3 = np.round(TrainPred2)
print(accuracy_score(TrainPred3, TrainBinary))

TestBinary = np.asarray(test_labels)
TestPred = TestPred[:,1]
TestPred = np.round(TestPred)
print(accuracy_score(TestPred, TestBinary))
