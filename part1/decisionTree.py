import copy
import operator
from math import log
from node import node
'''
def createTree(root, att, tempDataTable):
	#if root.result!=None:
	#	return root
	if len(att) == 0 or root.result == 'spam' or root.result == 'notspam': ##
		if root.positive > root.negative:
			root.result = 'spam'
		else:
			root.result = 'notspam'
		return root
	else:
		attribute = att[0]
		factor = getSplitFactor(attribute, tempDataTable)	#smaller values in left, larger in right subtree
		leftDataTable, rightDataTable = splitTrainingset(attribute, factor, tempDataTable)
		pos, neg = getPosNeg(tempDataTable)
		#root = node(attribute,factor,tempDataTable)
		root.createNode(attribute, factor, tempDataTable)
		att.remove(attribute)
		root.positive = pos		#positive for spam
		root.negative = neg		#negative for not spam
		if root.positive == 0:
			root.result = 'notspam'
		elif root.negative == 0:
			root.result = 'spam'
		root.leftNode = createTree(node(), att, leftDataTable)		#left subtree
		root.rightNode = createTree(node(), att, rightDataTable)	#right subtree
'''

def createTree(tempDataTable, attributes, root = None):

	# we will find two things-
	# 1. whether all the rows in tempDataTable are homogenous
	# 2. To prevent overfitting, if size of tempDataTable gets smaller than 50, then
	# declare the majority class as root.result
	if(homogenous(tempDataTable)):
		pos, neg = getPosNeg(tempDataTable)
		if pos > neg:
			root.result = 'spam'
		else:
			root.result = 'notspam'
		return root
	if (overfitting(tempDataTable)):
		root.result = getOverfittingResult(tempDataTable)
		return root
	attribute, factor = chooseAttribute(attributes, tempDataTable)
	leftDataTable, rightDataTable = splitTrainingset(attribute, attributes, factor, tempDataTable)
	pos, neg = getPosNeg(tempDataTable)
	root = node(attribute, factor, tempDataTable)
	root.positive = pos  # positive for spam
	root.negative = neg  # negative for not spam
	if root.positive == 0:
		root.result = 'notspam'
	elif root.negative == 0:
		root.result = 'spam'
	root.left = createTree(leftDataTable, attributes, root)
	root.right = createTree(rightDataTable, attributes, root)
	return root

def homogenous(tempDataTable):
	pos, neg = getPosNeg(tempDataTable)
	if pos ==0 or neg == 0:
		return True
	else:
		return False

def overfitting(tempDataTable):
	#TODO: change this to 50
	if len(tempDataTable) < 100:
		return True

def getOverfittingResult(tempDataTable):
	pos, neg = getPosNeg(tempDataTable)
	if pos>neg:
		return 'spam'
	else:
		return 'notspam'

def chooseAttribute(attributes, dataTable):
	#allFactors = []
	#igList = []
	'''
	for attr in attributes:
		s,e = getRange(attr, attributes, dataTable)
		factors = findAllFactors(s, e)
		igf = []
		for f in factors:
			i = informationGain(f, attr, attributes, dataTable)
			igf.append(i)
		ind = igf.index(max(igf))
		igList.append(igf[ind])
		allFactors.append(factors[ind])
	igIndex = igList.index(max(igList))
	#TODO: Correct the mistake here. Take minimum entropy instead of maximum
	#factIndex = allFactors.index(max(entropies))
	maxFactor = allFactors[igIndex]
	splitAttribute = attributes[igIndex]
	return splitAttribute, maxFactor'''

	bestThresholds = []
	bestGains = []
	for j in range(len(dataTable[0])-1):
		w = attributes[j]
		L = [row[j] for row in dataTable]	
		s,e = min(L),max(L)
		thresholds = findAllFactors(s,e)
		gains = []
		for t in thresholds:
			gains.append(findInformationGain(t,j,attributes,dataTable)) 
		max(gains)
		max_index, max_value = max(enumerate(gains), key=operator.itemgetter(1))
		bestGains.append(max_value)
		bestThresholds.append(thresholds[max_index])

	max_index, max_value = max(enumerate(bestGains), key=operator.itemgetter(1))
	finalThreshold = bestThresholds[max_index]
	return attributes[max_index],finalThreshold

def getRange(attr, attributes, dataTable):
	index = attributes.index(attr)
	newlist = [item[index] for item in dataTable]
	low = min(newlist)
	high = max(newlist)
	return low, high

def findAllFactors(low, high):
	
	if (low>high):
		raise Exception('Low can not be higher than high ')
	if low == high:
		factors = []
		factors.append(low)
		return factors
	val = (high-low)/4
	factors = []
	item = low
	for i in range(1, 4):
		item += val
		factors.append(item)
	return factors

def getEntropy(dataTable):
	pos,neg = getPosNeg(dataTable)
	if pos == 0 or neg == 0:
		return 0.0
	total = float(pos+neg)
	return -(pos/total)*log(pos/total,2) - (neg/total)*log(neg/total,2)

def findInformationGain(threshold, wordIndex, attributes, dataTable):
	wholeColumn = [ row[wordIndex] for row in dataTable]
	less = []
	more = []
	for i in range(len(wholeColumn)):
		if wholeColumn[i] <= threshold:
			less.append(dataTable[i])
		else:
			more.append(dataTable[i])

	e,e1,e2 = getEntropy(dataTable),getEntropy(less),getEntropy(more)
	w1 = float(len(less))/len(dataTable)
	w2 = float(len(more))/len(dataTable)
	gain = e - (e1*w1 + e2*w2)
	return gain

def informationGain(factor, attr, attributes, dataTable):
	index = attributes.index(attr)
	newlist = [item[index] for item in dataTable]
	less = float(len([1 for i in newlist if i < factor]))
	more = float(len([1 for i in newlist if i >= factor]))
	if (less+more) == 0:
		return 0
	else:
		pless = float(less/(less+more))
		pmore = float(more/(less+more))
		entropy = (-1)*(pless*log(pless if pless > 0 else 1)) - (pmore*log(pmore if pmore>0 else 1))
		#TODO: Ask how to calculate information gain using entropy over all the attributes
		return entropy

#gets the index of the attribute in the table. For now it is always the first attribute,
#but later, the attribute will be decided based on entropy
def getIndex(attribute, table):
	for i, j in enumerate(table):
		if attribute in j:
			return (i, j.index(attribute))

#calculates the positive and negative count in order to calculate the confidence factor
def getPosNeg(tempDataTable):
	pos = 0
	neg = 0
	#classArray = transpose(tempDataTable)[0]
	for i in range (0,len(tempDataTable)):
		if tempDataTable[i][len(tempDataTable[0]) - 1] == 1:
			pos += 1
		else:
			neg += 1
	return pos, neg

#this function creates two subtables with values for the attribute less than and greater than
#the splitting factor
def splitTrainingset(attribute, attributes, value, tempDataTable):
	leftDataTable = copy.deepcopy(tempDataTable)
	rightDataTable = copy.deepcopy(tempDataTable)
	index = attributes.index(attribute)
	for i in range (0, len(tempDataTable)):
		if tempDataTable[i][index] > value:
			leftDataTable.remove(tempDataTable[i])
		else:
			rightDataTable.remove(tempDataTable[i])
	return leftDataTable, rightDataTable

#traversing the data tree to test the document as spam or notspam
def traverseTree(root, dataRow, attributes):
	temp = root

	while temp.result == None:
		tempAttr = temp.attribute
		tempFactor = temp.factor
		attrIndex = attributes.index(tempAttr)
		dataRowValue = dataRow[attrIndex]
		if dataRowValue <= tempFactor:
			temp = temp.left
		else:
			temp = temp.right
	return temp.result
