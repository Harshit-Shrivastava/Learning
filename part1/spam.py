import math
import sys
import os
import operator
import string
import copy
import numpy as np
import pickle
from decisionTree import createTree, traverseTree
from binaryDecisionTree import decisionTree
from node import node
epsilon = 0.0000000001
dividend = 250

def trainNaiveBayes(data_directory,model):
	#print 'Hello-word!,'.translate(None, string.punctuation)
	nonspamDict = {}
	spamDict = {}

	print('Training the Naive Bayes Classifier... Please wait')
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
	
	
	modelFile = open(model, 'w')
	for k in set(spamDict.keys())|set(nonspamDict.keys()):
		if k not in ('',' ','the','to'):
			v1 = spamDict.get(k,epsilon)
			v2 = nonspamDict.get(k,epsilon)
			line = k +","+str(v1)+","+str(v2)
			modelFile.write(line)
			modelFile.write('\n')
	modelFile.close()
	print('Training complete. The model is saved on %s'%(model))
def testNaiveBayes(data_directory,model):
	nonspamDict = {}
	spamDict = {}
	print('Loading the model from %s'%(model))
	lines = [line.rstrip('\n') for line in open(model)]
	for line in lines:
		k,v1,v2 = line.split(",")
		spamDict[k] = float(v1)
		nonspamDict[k] = float(v2)
	
	print('Testing the model...')
	success = 0.0
	failure = 0.0
	TP,FP,TN,FN = 0.0,0.0,0.0,0.0
	for fname in os.listdir(data_directory+'/spam'):
		spamSum = 0.0
		nonspamSum = 0.0
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
		spamSum = 0.0
		nonspamSum = 0.0
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

def trainDecisionTree(data_directory,model):
	nonspamDict = {}
	spamDict = {}
	print('Training the Decision Tree Classifier... Please wait')
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


def testDecisionTree(data_directory,model):
	nonspamDict = {}
	spamDict = {}
	print('Testing the Decision Tree Classifier... Please wait')
	for fname in os.listdir(data_directory + '/spam'):
		with open(data_directory + '/spam/' + fname, 'r') as f:
			lines = f.readlines()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					# if word == '':
					#	raw_input()
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
	#print len(dataTable)
	#print len(dataTable[0])
	#print attributes
	# print attributeSet
	#print dataTable[0]
	print 'Reading decision tree from memory'
	root = pickle.load(open(model, 'rb'))
	#print (root)
	print 'Claasifying the emails'
	correct = 0
	spam = 0
	notspam = 0
	for i in range (0, len(dataTable)):
		dataTableRow = dataTable[i]
		#print('%d. Acutal label = %s and Predicted Label = ')
		if dataTableRow[len(dataTableRow) -1] == 1:
			actualResult = 'spam'
		else:
			actualResult = 'notspam'
		predictedResult = traverseTree(root, dataTableRow, attributes)
		if predictedResult == 'spam':
			spam += 1
		else:
			notspam += 1
		if actualResult == predictedResult:
			#print('%d. Acutal label = %s and Predicted Label = %s and its a success'%(i,actualResult,predictedResult))
			correct += 1
		#else:
			#print('%d. Acutal label = %s and Predicted Label = %s and its a failure' % (i, actualResult, predictedResult))
	print 'Accuracy %f' %(float(correct)/len(dataTable))
	print 'Correctly classified %d emails out of %d' % (correct, len(dataTable))
	print 'Clasified %d emails as spam and %d emails as not-spam' % (spam, notspam)

def trainBinaryDecisionTree(data_directory, model):
	nonspamDict = {}
	spamDict = {}
	print('Training the Decision Tree Classifier... Please wait')
	for fname in os.listdir(data_directory + '/spam'):
		with open(data_directory + '/spam/' + fname, 'r') as f:
			lines = f.readlines()
		for line in lines:
			for w in line.strip().split(' '):
				word = w.lower()
				# if word == '':
				#	raw_input()
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
	print len(dataTable)
	print len(dataTable[0])
	print attributes
	# print attributeSet
	print dataTable[0]
	# build the decision tree
	# write the decision tree model on the filename passed as 'model' so that later we can load the model and test new data instances
	# creating the decision tree
	# recursively create the decision tree
	root = decisionTree(dataTable, attributes, root=None)
	print root
	# save the decision tree into the model file
	with open(model, 'wb') as output:
		pickle.dump(root, output, pickle.HIGHEST_PROTOCOL)
	print 'Decision tree saved to memory'

def testBinaryDecisionTree(data_directory, model):
	nonspamDict = {}
	spamDict = {}
	print('Testing Binary Decision Tree Classifier... Please wait')
	for fname in os.listdir(data_directory + '/spam'):
		with open(data_directory + '/spam/' + fname, 'r') as f:
			lines = f.readlines()
			for line in lines:
				for w in line.strip().split(' '):
					word = w.lower()
					# if word == '':
					#	raw_input()
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
							#else:
							#	wordDict[word] += 1

			for w in attributes:
				datarow.append(wordDict.get(w, 0))
			datarow.append(0)  # 1 for spam, 0 for not spam
			dataTable.append(datarow)

	# each attributes is a word which is stored on the list called 'attributes'
	# so, dataTable has 2646 rows and each row has 7417 attributes where the last attribute is target attribute
	print len(dataTable)
	print len(dataTable[0])
	print attributes
	# print attributeSet
	print dataTable[0]
	print 'Reading decision tree from memory'
	root = pickle.load(open(model, 'rb'))
	print (root)
	correct = 0
	spam = 0
	notspam = 0
	for i in range(0, len(dataTable)):
		dataTableRow = dataTable[i]
		if dataTableRow[len(dataTableRow) - 1] == 1:
			actualResult = 'spam'
		else:
			actualResult = 'notspam'
		predictedResult = traverseTree(root, dataTableRow, attributes)
		if predictedResult == 'spam':
			spam += 1
		else:
			notspam += 1
		if actualResult == predictedResult:
			correct += 1
	print 'Accuracy'
	print correct
	print float(correct) / len(dataTable)
	print spam
	print notspam

mode,technique,data_directory,model = sys.argv[1:]
if mode == 'train':
	if technique == 'bayes':
		trainNaiveBayes(data_directory,model)
	else:
		trainDecisionTree(data_directory,model)
		#trainBinaryDecisionTree(data_directory, model)
else:
	if technique == 'bayes':
		testNaiveBayes(data_directory,model)
	else:
		testDecisionTree(data_directory,model)
		#testBinaryDecisionTree(data_directory, model)