import math
import sys
import os
import operator
import string
epsilon = 0.0000000001
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
					if word not in ('','the','to'):
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
					if word not in ('','the','to'):
						word = word.translate(None, string.punctuation)
						if nonspamDict.get(word,None)==None:
							nonspamDict[word] = 1.0
						else:
							nonspamDict[word] += 1.0
	
	s = sum([v for (k,v) in spamDict.items()])
	for (k,v) in spamDict.items():
		spamDict[k] = v/s
	s = sum([v for (k,v) in nonspamDict.items()])
	for (k,v) in nonspamDict.items():
		nonspamDict[k] = v/s
	
	
	modelFile = open(model, 'w')
	for k in set(spamDict.keys())|set(nonspamDict.keys()):
		if k not in ('','the','to'):
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
					spamSum += (-math.log10(a))
					a = nonspamDict.get(word,epsilon)
					nonspamSum += (-math.log10(a))
		 
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
					spamSum += (-math.log10(a))
					a = nonspamDict.get(word,epsilon)
					nonspamSum += (-math.log10(a))
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
	#print os.listdir(data_directory)
	a = 0

def testDecisionTree(data_directory,model):
	#print os.listdir(data_directory)
	a = 0


mode,technique,data_directory,model = sys.argv[1:]
if mode == 'train':
	if technique == 'bayes':
		trainNaiveBayes(data_directory,model)
	else:
		trainDecisionTree(data_directory,model)
else:
	if technique == 'bayes':
		testNaiveBayes(data_directory,model)
	else:
		testDecisionTree(data_directory,model)