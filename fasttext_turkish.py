#!/usr/bin/python

import fasttext

filename = '/Users/szeybek/Documents/trainingData_s.txt';

# Skipgram model
model = fasttext.skipgram(filename, 'model')
print(model.words)# list of words in dictionary

print(model['asker'])


classifier = fasttext.supervised(filename, 'model',label_prefix='__label__')
result = classifier.test(filename)
print ('For fasttext skipgram model P@1:', result.precision)
print ('For fasttext skipgram model R@1:', result.recall)

# CBOW model
model = fasttext.cbow(filename, 'model',)
print (model.words) # list of words in dictionary


print(model['asker'])

classifier = fasttext.supervised(filename, 'model',label_prefix='__label__')
result = classifier.test(filename)
print ('For fasttext CBOW modelP@1:', result.precision)
print ('For fasttext CBOW modelR@1:', result.recall)
print ('Number of examples:', result.nexamples)
