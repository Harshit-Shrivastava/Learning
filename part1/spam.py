## We have considered and used both binary and continuous freqeuncy of words for naive bayes classifier and decision tree classifier. Binary frequency corresponds
## to settings where we only considered whether a particular word appeared in a document or not and Continuous frequency corresponds to settings where we considered
# number of times any partical appeared in a document. The accuracies are reported below-
## 1.1 Naive Bayes with Binary Frequency     : 98%
## 1.2 Naive Bayes with Continuous Frequency : 95%
## 2.1 Decision Tree with Binary Features    : 98%
## 2.2 Decision Tree with Continuous Features: 98%
##
## Among all the techniques, 'Binary Frequency Distribution' worked best for us for the given
## dataset.
					## ### Naive Bayes ### ##
# We calculated the likelihood probabilites P(word = 'apple'|Spam = 'Y') and P(word = 'apple'|Spam = 'N') from the training set. We also calculated the class
# Priors: P(Spam = 'Y') and P(Spam = 'N'). We used odds ratio to predict the label of a document and also we have set the threshold of odds ratio to be 1. So,
# if odds ratio is greater than 1, we label the document as 'spam' otherwise 'notspam'.

#					## ### Decision Tree ### ##
# Continuous frequency distibution:-
# For decision tree with continuous frequency distribution, while training, as Professor Crandall
# explained in the class and during the assignment discussions at the start of the class, we took
# the occurrence of each word in the document as an individual event and calculated a continuous
# frequency distribution table based on it. In this table, to find the word that we should use to
# split the tree on, we used information gain (highest), which was calculated using entropies. We
# calculated information gain at various intervals for the distribution of a word to calculate
# the information gain at of those intervals (which we called split factors) for each word.
# Highest among all information gain, and the corresponding word and split factor were chosen to
# create a node. All the training data set having frequency distribution less than or equal to the
# split factor for that word on that node went into the left subtree and remainig into the right
# subtree.
# Binary frequency distribution:-
# The occurrence of a word in a document was given a binary value of 0 or 1, as in, if the word
# occurs in a document, it gets a distribution 1, 0 otherwise. To select the word to use to
# split the decision tree next, information gain was used. Word with the highest information gain
# was chosen to split the training data set at this node. All the training data set having
# frequency distribution 0 for this word formed the left subtree and the remaining (that has 1)
# formed the right subtree.
# Conditions verified while training:-
# 1. Homogeneous data set
# 2. Overfitting
# 3. If all words have been observed (for binary distribution, for continuous distribution, the
#    word can be observed again)
# Confusion matrix is printed where TP denotes true positive (a spam was classified as a spam),
# FN denotes true negative (a notspam is classified as notspam), FP denotes false positive (a
# notspam was classified as a spam and TN denotes false negative (a spam was classified a notspam)

###### ===========================================================================================
#       Choosing between continuous and binary classifier:-
###### ===========================================================================================

# The variable 'featureType' in spam.py  (declared and defined just below the imports)
# denotes whether the classification is happening over binary distribution or continuous
# distribution
#  1 denotes continuous features
#  2 denotes binary features

import math
import sys
import os
import operator
import string
import copy
import numpy as np
import pickle
from decisionTree import createTree, traverseTree, printDecisionTree
from binaryDecisionTree import binaryDecisionTree, traverseBinaryTree, printBinaryDecisionTree
from node import node
epsilon = 0.0000000001
dividend = 30

#1 denotes continuous features
#2 denotes binary features
featureType = 2
def trainContinuousNaiveBayes(data_directory,model):
	nonspamDict = {}
	spamDict = {}
	spamDocCount, nonSpamDocCount = 0,0

	spamDocCount = len(os.listdir(data_directory+'/spam'))
	nonSpamDocCount = len(os.listdir(data_directory+'/notspam'))
	print('Training the Naive Bayes Classifier with Continuous features... Please wait')
	for fname in os.listdir(data_directory+'/spam'):
		with open(data_directory+'/spam/'+fname,'r') as f:
    			lines = f.readlines()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('',' ','the','to'):
						word = word.translate(None,string.punctuation)
						if spamDict.get(word,None)==None:
							spamDict[word] = 1.0
						else:
							spamDict[word] += 1.0

	for fname in os.listdir(data_directory+'/notspam'):
		with open(data_directory+'/notspam/'+fname,'r') as f:
    			lines = f.readlines()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('',' ','the','to'):
						word = word.translate(None, string.punctuation)
						if nonspamDict.get(word,None)==None:
							nonspamDict[word] = 1.0
						else:
							nonspamDict[word] += 1.0
	
	for w in set(spamDict.keys())|set(nonspamDict.keys()):
		v1 = spamDict.get(w,0.0)
		v2 = nonspamDict.get(w,0.0)
		spamDict[w] = v1/(v1+v2);
		nonspamDict[w] = v2/(v1+v2);
	#s = sum([v for (k,v) in spamDict.items()])
	#for (k,v) in spamDict.items():
	#	spamDict[k] = v/s
	#s = sum([v for (k,v) in nonspamDict.items()])
	#for (k,v) in nonspamDict.items():
	#	nonspamDict[k] = v/s
	
	
	probSpam = spamDocCount/float(spamDocCount+nonSpamDocCount)
	probNonSpam = nonSpamDocCount/float(spamDocCount+nonSpamDocCount)
	posterior = {}
	for w in spamDict.keys():
		v1 = spamDict.get(w,epsilon)*probSpam
		v2 = nonspamDict.get(w,epsilon)*probNonSpam
		posterior[w] = v1/(v1+v2)
	
	L = sorted(posterior.items(), key=operator.itemgetter(1), reverse=True)
	print L[0:10]
	
	modelFile = open(model, 'w')
	modelFile.write(str(spamDocCount)+","+str(nonSpamDocCount))
	modelFile.write('\n')
	for k in set(spamDict.keys())|set(nonspamDict.keys()):
		if k not in ('',' ','the','to'):
			v1 = spamDict.get(k,epsilon)
			v2 = nonspamDict.get(k,epsilon)
			line = k +","+str(v1)+","+str(v2)
			modelFile.write(line)
			modelFile.write('\n')
	modelFile.close()
	print('Training complete. The model is saved on %s'%(model))


def trainBinaryNaiveBayes(data_directory,model):
	nonspamDict = {}
	spamDict = {}
	spamDocCount, nonSpamDocCount = 0,0

	spamDocCount = len(os.listdir(data_directory+'/spam'))
	nonSpamDocCount = len(os.listdir(data_directory+'/notspam'))
	print('Training the Naive Bayes Classifier with Binary features... Please wait')
	for fname in os.listdir(data_directory+'/spam'):
		with open(data_directory+'/spam/'+fname,'r') as f:
    			lines = f.readlines()
			wordSet = set()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('',' ','the','to'):
						word = word.translate(None,string.punctuation)
						wordSet.add(word)
						#if spamDict.get(word,None)==None:
						#	spamDict[word] = 1.0
						#else:
						#	spamDict[word] += 1.0
			for w in wordSet:
				if spamDict.get(w,None)==None:
					spamDict[w] = 1.0
				else:
					spamDict[w] += 1.0

	for fname in os.listdir(data_directory+'/notspam'):
		with open(data_directory+'/notspam/'+fname,'r') as f:
    			lines = f.readlines()
			wordSet = set()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('',' ','the','to'):
						word = word.translate(None, string.punctuation)
						wordSet.add(word)
						#if nonspamDict.get(word,None)==None:
						#	nonspamDict[word] = 1.0
						#else:
						#	nonspamDict[word] += 1.0
			for w in wordSet:
				if nonspamDict.get(w,None)==None:
					nonspamDict[w] = 1.0
				else:
					nonspamDict[w] += 1.0
	

	for (k,v) in spamDict.items():
		spamDict[k] = v/spamDocCount

	for (k,v) in nonspamDict.items():
		nonspamDict[k] = v/nonSpamDocCount

	'''
	for w in set(spamDict.keys())|set(nonspamDict.keys()):
		v1 = spamDict.get(w,0.0)
		v2 = nonspamDict.get(w,0.0)
		spamDict[w] = v1/(v1+v2);
		nonspamDict[w] = v2/(v1+v2)
	'''
	#s = sum([v for (k,v) in spamDict.items()])
	#for (k,v) in spamDict.items():
	#	spamDict[k] = v/s
	#s = sum([v for (k,v) in nonspamDict.items()])
	#for (k,v) in nonspamDict.items():
	#	nonspamDict[k] = v/s
	
	
	probSpam = spamDocCount/float(spamDocCount+nonSpamDocCount)
	probNonSpam = nonSpamDocCount/float(spamDocCount+nonSpamDocCount)
	posterior = {}
	for w in spamDict.keys():
		v1 = spamDict.get(w,epsilon)*probSpam
		v2 = nonspamDict.get(w,epsilon)*probNonSpam
		posterior[w] = v1/(v1+v2)
	
	L = sorted(posterior.items(), key=operator.itemgetter(1), reverse=True)
	print L[0:10]
	
	modelFile = open(model, 'w')
	modelFile.write(str(spamDocCount)+","+str(nonSpamDocCount))
	modelFile.write('\n')
	for k in set(spamDict.keys())|set(nonspamDict.keys()):
		if k not in ('',' ','the','to'):
			v1 = spamDict.get(k,epsilon)
			v2 = nonspamDict.get(k,epsilon)
			line = k +","+str(v1)+","+str(v2)
			modelFile.write(line)
			modelFile.write('\n')
	modelFile.close()
	print('Training complete. The model is saved on %s'%(model))

def testBinaryNaiveBayes(data_directory,model):
	nonspamDict = {}
	spamDict = {}
	print('Loading the model with binary features from %s'%(model))
	lines = [line.rstrip('\n') for line in open(model)]
	line = lines[0]
	spamDocCount , nonSpamDocCount = int(line.split(",")[0]),int(line.split(",")[1])

	for i in xrange(1,len(lines)):
		line = lines[i]
		k,v1,v2 = line.split(",")
		spamDict[k] = float(v1)
		nonspamDict[k] = float(v2)
	
	print('Testing the model...')
	success = 0.0
	failure = 0.0
	TP,FP,TN,FN = 0.0,0.0,0.0,0.0
	for fname in os.listdir(data_directory+'/spam'):
		spamSum = -math.log10(spamDocCount/float(spamDocCount+nonSpamDocCount))
		nonspamSum = -math.log10(nonSpamDocCount/float(spamDocCount+nonSpamDocCount))
		with open(data_directory+'/spam/'+fname,'r') as f:
    			lines = f.readlines()
			wordSet = set()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					word = word.translate(None, string.punctuation)
					wordSet.add(word)
					#a = spamDict.get(word,epsilon)
					#spamSum += (-math.log10(a if a!=0.0 else epsilon))
					#a = nonspamDict.get(word,epsilon)
					#nonspamSum += (-math.log10(a if a!=0.0 else epsilon))
			for word in wordSet:
				a = spamDict.get(word,epsilon)
				spamSum += (-math.log10(a if a!=0.0 else epsilon))
				a = nonspamDict.get(word,epsilon)
				nonspamSum += (-math.log10(a if a!=0.0 else epsilon))
		if spamSum<nonspamSum:
			success += 1.0
			TP += 1.0
		else:
			failure += 1.0
			TN += 1.0
	for fname in os.listdir(data_directory+'/notspam'):
		spamSum = -math.log10(spamDocCount/float(spamDocCount+nonSpamDocCount))
		nonspamSum = -math.log10(nonSpamDocCount/float(spamDocCount+nonSpamDocCount))
		with open(data_directory+'/notspam/'+fname,'r') as f:
    			lines = f.readlines()
			wordSet = set()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					word = word.translate(None, string.punctuation)
					wordSet.add(word)
					#a = spamDict.get(word,epsilon)
					#spamSum += (-math.log10(a if a!=0.0 else epsilon))
					#a = nonspamDict.get(word,epsilon)
					#nonspamSum += (-math.log10(a if a!=0.0 else epsilon))
			for word in wordSet:
				a = spamDict.get(word,epsilon)
				spamSum += (-math.log10(a if a!=0.0 else epsilon))
				a = nonspamDict.get(word,epsilon)
				nonspamSum += (-math.log10(a if a!=0.0 else epsilon))
		if spamSum>nonspamSum:
			success += 1.0
			FN += 1.0;
		else:
			failure += 1.0
			FP += 1.0
	accuracy = success/(success+failure)
	print('Accuracy: %f' %(accuracy))
	print 'Confusion Matrix: '
	print(' TP = %-20.0f TN = %-20.0f'%(TP,TN))
	print(' FP = %-20.0f FN = %-20.0f'%(FP,FN))

def testContinuousNaiveBayes(data_directory,model):
	nonspamDict = {}
	spamDict = {}
	print('Loading the model with continuous features from %s'%(model))
	lines = [line.rstrip('\n') for line in open(model)]
	line = lines[0]
	spamDocCount , nonSpamDocCount = int(line.split(",")[0]),int(line.split(",")[1])

	for i in xrange(1,len(lines)):
		line = lines[i]
		k,v1,v2 = line.split(",")
		spamDict[k] = float(v1)
		nonspamDict[k] = float(v2)
	
	print('Testing the model...')
	success = 0.0
	failure = 0.0
	TP,FP,TN,FN = 0.0,0.0,0.0,0.0
	for fname in os.listdir(data_directory+'/spam'):
		spamSum = -math.log10(spamDocCount/float(spamDocCount+nonSpamDocCount))
		nonspamSum = -math.log10(nonSpamDocCount/float(spamDocCount+nonSpamDocCount))
		with open(data_directory+'/spam/'+fname,'r') as f:
    			lines = f.readlines()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					#word = word.translate(None, string.punctuation)
					a = spamDict.get(word,epsilon)
					spamSum += (-math.log10(a if a!=0.0 else epsilon))
					a = nonspamDict.get(word,epsilon)
					nonspamSum += (-math.log10(a if a!=0.0 else epsilon))
		 
		if spamSum<nonspamSum:
			success += 1.0
			TP += 1.0
		else:
			failure += 1.0
			TN += 1.0
	for fname in os.listdir(data_directory+'/notspam'):
		spamSum = -math.log10(spamDocCount/float(spamDocCount+nonSpamDocCount))
		nonspamSum = -math.log10(nonSpamDocCount/float(spamDocCount+nonSpamDocCount))
		with open(data_directory+'/notspam/'+fname,'r') as f:
    			lines = f.readlines()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					#word = word.translate(None, string.punctuation)
					a = spamDict.get(word,epsilon)
					spamSum += (-math.log10(a if a!=0.0 else epsilon))
					a = nonspamDict.get(word,epsilon)
					nonspamSum += (-math.log10(a if a!=0.0 else epsilon))
		if spamSum>nonspamSum:
			success += 1.0
			FN += 1.0;
		else:
			failure += 1.0
			FP += 1.0
	accuracy = success/(success+failure)
	print('Accuracy: %f' %(accuracy))
	print 'Confusion Matrix: '
	print(' TP = %-20.0f TN = %-20.0f'%(TP,TN))
	print(' FP = %-20.0f FN = %-20.0f'%(FP,FN))

'''Function to train decision tree classifier based on  continuos frequency distribution'''
def trainDecisionTree(data_directory,model):
	nonspamDict = {}
	spamDict = {}
	print('Training decision tree classifier based on continuous frequency distribution... Please wait')
	for fname in os.listdir(data_directory+'/spam'):
		with open(data_directory+'/spam/'+fname,'r') as f:
    			lines = f.readlines()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					#if word == '':
					#	raw_input()
					if word not in ('',' ','the','to'):
						word = word.translate(None,string.punctuation)
						if spamDict.get(word,None)==None:
							spamDict[word] = 1.0
						else:
							spamDict[word] += 1.0

	for fname in os.listdir(data_directory+'/notspam'):
		with open(data_directory+'/notspam/'+fname,'r') as f:
    			lines = f.readlines()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('',' ','the','to'):
						word = word.translate(None, string.punctuation)
						if nonspamDict.get(word,None)==None:
							nonspamDict[word] = 1.0
						else:
							nonspamDict[word] += 1.0

	b1 = sorted(spamDict.items(), key=operator.itemgetter(1), reverse=True)[:len(spamDict)/dividend]
	b2 = sorted(nonspamDict.items(), key=operator.itemgetter(1), reverse=True)[:len(nonspamDict)/dividend]

	union_set = set(b1).union(set(b2))
	attributes = [w[0] for w in union_set]
	if '' in attributes:
		attributes.remove('')

	attributeSet = set(attributes)

	dataTable = []

	for fname in os.listdir(data_directory+'/spam'):
		with open(data_directory+'/spam/'+fname,'r') as f:
    			lines = f.readlines()
			datarow = []
			wordDict = {}
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('',' ','the','to'):
						word = word.translate(None,string.punctuation)
						if word in attributeSet:
							if wordDict.get(word,None)==None:
								wordDict[word] = 1
							else:
								wordDict[word] += 1
			for w in attributes:
				datarow.append(wordDict.get(w,0))
			datarow.append(1) # 1 for spam, 0 for not spam
			dataTable.append(datarow)

	for fname in os.listdir(data_directory+'/notspam'):
		with open(data_directory+'/notspam/'+fname,'r') as f:
    			lines = f.readlines()
			datarow = []
			wordDict = {}
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('',' ','the','to'):
						word = word.translate(None,string.punctuation)
						if word in attributeSet:
							if wordDict.get(word,None)==None:
								wordDict[word] = 1
							else:
								wordDict[word] += 1

			for w in attributes:
				datarow.append(wordDict.get(w,0))
			datarow.append(0) # 1 for spam, 0 for not spam
			dataTable.append(datarow)

	# each attributes is a word which is stored on the list called 'attributes'
	# so, dataTable has 2646 rows and each row has 7417 attributes where the last attribute is target attribute
	#print len(dataTable)
	#print len(dataTable[0])
	#print attributes
	#print attributeSet
	#print dataTable[0]
	# build the decision tree
	# write the decision tree model on the filename passed as 'model' so that later we can load the model and test new data instances
	#creating the decision tree
	# recursively create the decision tree
	root = createTree(dataTable, attributes)
	print root
	#save the decision tree into the model file
	with open(model, 'wb') as output:
		pickle.dump(root, output, pickle.HIGHEST_PROTOCOL)
	print 'Decision tree saved to memory'

'''Function to test decision tree classifier based on continuos frequency distribution'''
def testDecisionTree(data_directory,model):
	nonspamDict = {}
	spamDict = {}
	print('Testing decision tree classifier based on continuous distrbution... Please wait')
	for fname in os.listdir(data_directory + '/spam'):
		with open(data_directory + '/spam/' + fname, 'r') as f:
			lines = f.readlines()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('', ' ', 'the', 'to'):
						word = word.translate(None, string.punctuation)
						if spamDict.get(word, None) == None:
							spamDict[word] = 1.0
						else:
							spamDict[word] += 1.0

	for fname in os.listdir(data_directory + '/notspam'):
		with open(data_directory + '/notspam/' + fname, 'r') as f:
			lines = f.readlines()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('', ' ', 'the', 'to'):
						word = word.translate(None, string.punctuation)
						if nonspamDict.get(word, None) == None:
							nonspamDict[word] = 1.0
						else:
							nonspamDict[word] += 1.0

	b1 = sorted(spamDict.items(), key=operator.itemgetter(1), reverse=True)[:len(spamDict) / dividend]
	b2 = sorted(nonspamDict.items(), key=operator.itemgetter(1), reverse=True)[:len(nonspamDict) / dividend]

	union_set = set(b1).union(set(b2))
	attributes = [w[0] for w in union_set]
	if '' in attributes:
		attributes.remove('')

	attributeSet = set(attributes)

	dataTable = []

	for fname in os.listdir(data_directory + '/spam'):
		with open(data_directory + '/spam/' + fname, 'r') as f:
			lines = f.readlines()
			datarow = []
			wordDict = {}
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('', ' ', 'the', 'to'):
						word = word.translate(None, string.punctuation)
						if word in attributeSet:
							if wordDict.get(word, None) == None:
								wordDict[word] = 1
							else:
								wordDict[word] += 1
			for w in attributes:
				datarow.append(wordDict.get(w, 0))
			datarow.append(1)  # 1 for spam, 0 for not spam
			dataTable.append(datarow)

	for fname in os.listdir(data_directory + '/notspam'):
		with open(data_directory + '/notspam/' + fname, 'r') as f:
			lines = f.readlines()
			datarow = []
			wordDict = {}
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('', ' ', 'the', 'to'):
						word = word.translate(None, string.punctuation)
						if word in attributeSet:
							if wordDict.get(word, None) == None:
								wordDict[word] = 1
							else:
								wordDict[word] += 1

			for w in attributes:
				datarow.append(wordDict.get(w, 0))
			datarow.append(0)  # 1 for spam, 0 for not spam
			dataTable.append(datarow)

	# each attributes is a word which is stored on the list called 'attributes'
	# so, dataTable has 2646 rows and each row has 7417 attributes where the last attribute is target attribute
	print 'Reading decision tree from memory'
	root = pickle.load(open(model, 'rb'))
	print 'Classifying emails as spams and not-spams'
	correct = 0
	spam = 0
	notspam = 0
	TP = 0
	TN = 0
	FN = 0
	FP = 0
	for i in range (0, len(dataTable)):
		dataTableRow = dataTable[i]
		if dataTableRow[len(dataTableRow) -1] == 1:
			actualResult = 'spam'
		else:
			actualResult = 'notspam'
		predictedResult = traverseTree(root, dataTableRow, attributes)
		if predictedResult == 'spam':
			spam += 1
			if predictedResult == actualResult:
				TP += 1
			else:
				FP += 1
		else:
			notspam += 1
			if predictedResult == actualResult:
				FN += 1
			else:
				TN += 1
		if actualResult == predictedResult:
			#print('%d. Acutal label = %s and Predicted Label = %s and its a success'%(i,actualResult,predictedResult))
			correct += 1
		#else:
			#print('%d. Acutal label = %s and Predicted Label = %s and its a failure' % (i, actualResult, predictedResult))
	print 'Accuracy %f' %(float(correct)/len(dataTable))
	print 'Correctly classified %d emails out of %d' % (correct, len(dataTable))
	print 'Clasified %d emails as spam and %d emails as not-spam' % (spam, notspam)
	print 'Confusion Matrix: '
	print(' TP = %-20.0f TN = %-20.0f' % (TP, TN))
	print(' FP = %-20.0f FN = %-20.0f' % (FP, FN))

	print 'Printing top nodes of the decision tree based on continuos frequency distribution'
	printDecisionTree(root)

'''Function to train decision tree constructed on binary distribution'''
def trainBinaryDecisionTree(data_directory, model):
	nonspamDict = {}
	spamDict = {}
	print('Training decision tree classifier based on binary distributions... Please wait')
	for fname in os.listdir(data_directory + '/spam'):
		with open(data_directory + '/spam/' + fname, 'r') as f:
			lines = f.readlines()
		for line in lines:
			for w in line.strip().split(' '):
				word = w.lower()
				if word not in ('', ' ', 'the', 'to'):
					word = word.translate(None, string.punctuation)
					if spamDict.get(word, None) == None:
						spamDict[word] = 1.0
					else:
						spamDict[word] += 1.0

	for fname in os.listdir(data_directory + '/notspam'):
		with open(data_directory + '/notspam/' + fname, 'r') as f:
			lines = f.readlines()
		for line in lines:
			for w in line.strip().split(' '):
				word = w.lower()
				if word not in ('', ' ', 'the', 'to'):
					word = word.translate(None, string.punctuation)
					if nonspamDict.get(word, None) == None:
						nonspamDict[word] = 1.0
					else:
						nonspamDict[word] += 1.0

	b1 = sorted(spamDict.items(), key=operator.itemgetter(1), reverse=True)[:len(spamDict) / dividend]
	b2 = sorted(nonspamDict.items(), key=operator.itemgetter(1), reverse=True)[:len(nonspamDict) / dividend]

	union_set = set(b1).union(set(b2))
	attributes = [w[0] for w in union_set]
	if '' in attributes:
		attributes.remove('')
	attributeSet = set(attributes)
	dataTable = []

	for fname in os.listdir(data_directory + '/spam'):
		with open(data_directory + '/spam/' + fname, 'r') as f:
			lines = f.readlines()
		datarow = []
		wordDict = {}
		for line in lines:
			for w in line.strip().split(' '):
				word = w.lower()
				if word not in ('', ' ', 'the', 'to'):
					word = word.translate(None, string.punctuation)
					if word in attributeSet:
						if wordDict.get(word, None) == None:
							wordDict[word] = 1
		for w in attributes:
			datarow.append(wordDict.get(w, 0))
		datarow.append(1)  # 1 for spam, 0 for not spam
		dataTable.append(datarow)


	for fname in os.listdir(data_directory + '/notspam'):
		with open(data_directory + '/notspam/' + fname, 'r') as f:
			lines = f.readlines()
		datarow = []
		wordDict = {}
		for line in lines:
			for w in line.strip().split(' '):
				word = w.lower()
				if word not in ('', ' ', 'the', 'to'):
					word = word.translate(None, string.punctuation)
					if word in attributeSet:
						if wordDict.get(word, None) == None:
							wordDict[word] = 1
		for w in attributes:
			datarow.append(wordDict.get(w, 0))
		datarow.append(0)  # 1 for spam, 0 for not spam
		dataTable.append(datarow)

	# each attributes is a word which is stored on the list called 'attributes'
	# so, dataTable has 2646 rows and each row has 7417 attributes where the last attribute is target attribute
	# build the decision tree
	# write the decision tree model on the filename passed as 'model' so that later we can load the model and test new data instances

	wordList = []
	for w in attributes:
		wordList.append(w)

	root = binaryDecisionTree(dataTable, attributes, wordList)
	print root
	# save the decision tree into the model file
	with open(model, 'wb') as output:
		pickle.dump(root, output, pickle.HIGHEST_PROTOCOL)
	print 'Decision tree saved to memory'

'''Function to test decision tree constructed on  binary distributions'''
def testBinaryDecisionTree(data_directory, model):
	nonspamDict = {}
	spamDict = {}
	print('Testing decision tree classifier with binary distributions... Please wait')
	for fname in os.listdir(data_directory + '/spam'):
		with open(data_directory + '/spam/' + fname, 'r') as f:
			lines = f.readlines()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('', ' ', 'the', 'to'):
						word = word.translate(None, string.punctuation)
						if spamDict.get(word, None) == None:
							spamDict[word] = 1.0
						else:
							spamDict[word] += 1.0

	for fname in os.listdir(data_directory + '/notspam'):
		with open(data_directory + '/notspam/' + fname, 'r') as f:
			lines = f.readlines()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('', ' ', 'the', 'to'):
						word = word.translate(None, string.punctuation)
						if nonspamDict.get(word, None) == None:
							nonspamDict[word] = 1.0
						else:
							nonspamDict[word] += 1.0

	b1 = sorted(spamDict.items(), key=operator.itemgetter(1), reverse=True)[:len(spamDict) / dividend]
	b2 = sorted(nonspamDict.items(), key=operator.itemgetter(1), reverse=True)[:len(nonspamDict) / dividend]

	union_set = set(b1).union(set(b2))
	attributes = [w[0] for w in union_set]
	if '' in attributes:
		attributes.remove('')
	attributeSet = set(attributes)
	dataTable = []

	for fname in os.listdir(data_directory + '/spam'):
		with open(data_directory + '/spam/' + fname, 'r') as f:
			lines = f.readlines()
			datarow = []
			wordDict = {}
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('', ' ', 'the', 'to'):
						word = word.translate(None, string.punctuation)
						if word in attributeSet:
							if wordDict.get(word, None) == None:
								wordDict[word] = 1
							#else:
							#	wordDict[word] += 1
			for w in attributes:
				datarow.append(wordDict.get(w, 0))
			datarow.append(1)  # 1 for spam, 0 for not spam
			dataTable.append(datarow)

	for fname in os.listdir(data_directory + '/notspam'):
		with open(data_directory + '/notspam/' + fname, 'r') as f:
			lines = f.readlines()
			datarow = []
			wordDict = {}
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					if word not in ('', ' ', 'the', 'to'):
						word = word.translate(None, string.punctuation)
						if word in attributeSet:
							if wordDict.get(word, None) == None:
								wordDict[word] = 1
							#else:
							#	wordDict[word] += 1

			for w in attributes:
				datarow.append(wordDict.get(w, 0))
			datarow.append(0)  # 1 for spam, 0 for not spam
			dataTable.append(datarow)

	# each attributes is a word which is stored on the list called 'attributes'
	# so, dataTable has 2646 rows and each row has 7417 attributes where the last attribute is target attribute
	print 'Reading decision tree from memory'
	root = pickle.load(open(model, 'rb'))
	print 'Classifying emails as spams and not-spams'
	correct = 0
	spam = 0
	notspam = 0
	TP = 0
	TN = 0
	FP = 0
	FN = 0
	for i in range(0, len(dataTable)):
		dataTableRow = dataTable[i]
		# print('%d. Acutal label = %s and Predicted Label = ')
		if dataTableRow[len(dataTableRow) - 1] == 1:
			actualResult = 'spam'
		else:
			actualResult = 'notspam'
		predictedResult = traverseBinaryTree(root, dataTableRow, attributes)
		if predictedResult == 'spam':
			spam += 1
			if predictedResult == actualResult:
				TP += 1
			else:
				FP += 1
		else:
			notspam += 1
			if predictedResult == actualResult:
				FN += 1
			else:
				TN += 1
		if actualResult == predictedResult:
			# print('%d. Acutal label = %s and Predicted Label = %s and its a success'%(i,actualResult,predictedResult))
			correct += 1
		# else:
		# print('%d. Acutal label = %s and Predicted Label = %s and its a failure' % (i, actualResult, predictedResult))
	print 'Accuracy %f' % (float(correct) / len(dataTable))
	print 'Correctly classified %d emails out of %d' % (correct, len(dataTable))
	print 'Clasified %d emails as spam and %d emails as not-spam' % (spam, notspam)
	print(' TP = %-20.0f TN = %-20.0f' % (TP, TN))
	print(' FP = %-20.0f FN = %-20.0f' % (FP, FN))
	print 'Printing top nodes of the binary decision tree'
	printBinaryDecisionTree(root)

mode,technique,data_directory,model = sys.argv[1:]
if mode == 'train':
	if technique == 'bayes':
		if featureType==1:
			trainContinuousNaiveBayes(data_directory,model)
		else:
			trainBinaryNaiveBayes(data_directory,model)
	else:
		if (featureType == 1):
			# decision tree classifier based on continuous frequency distribution
			trainDecisionTree(data_directory,model)
		else:
			# decision tree classifier based on binary distribution
			trainBinaryDecisionTree(data_directory, model)
else:
	if technique == 'bayes':

		if featureType==1:
			testContinuousNaiveBayes(data_directory,model)
		else:
			testBinaryNaiveBayes(data_directory,model)

	else:
		if (featureType == 1):
			#decision tree classifier based on continuous frequency distribution
			testDecisionTree(data_directory,model)
		else:
			#decision tree classifier based on binary distribution
			testBinaryDecisionTree(data_directory, model)
