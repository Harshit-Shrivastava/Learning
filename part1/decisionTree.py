import operator
from math import log
from node import node

overfittingFactor = 40
thresholdCount = 2

def createTree(tempDataTable, attributes):
	if(homogenous(tempDataTable)):
		pos, neg = getPosNeg(tempDataTable)
		root = node(None,None,None)
		if pos > neg:
			root.result = 'spam'
		else:
			root.result = 'notspam'
		return root
	if (overfitting(tempDataTable)):
		root = node(None, None, None)
		root.result = getOverfittingResult(tempDataTable)
		return root
	attribute, factor = chooseAttribute(attributes, tempDataTable)
	#rows having value of the attribute <= factor form the left sub-tree,
	#rows having value of the attribute > factor form the right of the tree
	leftDataTable, rightDataTable = splitTrainingset(attribute, attributes, factor, tempDataTable)
	pos, neg = getPosNeg(tempDataTable)		#count of spams and notspams
	root = node(attribute, factor, tempDataTable)	#create the node with reqd info
	root.positive = pos  # positive for spam
	root.negative = neg  # negative for not spam
	root.left = createTree(leftDataTable, attributes)		#recursively create left subtree
	root.right = createTree(rightDataTable, attributes)	#recursively create right subtree
	return root

def chooseAttribute(attributes, dataTable):
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
	val = (high-low)/thresholdCount
	factors = []
	item = low
	for i in range(1, thresholdCount):
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

#calculates the positive and negative count in order to calculate the confidence factor
def getPosNeg(tempDataTable):
	pos = 0
	neg = 0
	for i in range (0,len(tempDataTable)):
		if tempDataTable[i][len(tempDataTable[0]) - 1] == 1:
			pos += 1
		else:
			neg += 1
	return pos, neg

#this function creates two subtables with values for the attribute less than and greater than
#the splitting factor
def splitTrainingset(attribute, attributes, value, tempDataTable):
	leftDataTable = []
	rightDataTable = []
	index = attributes.index(attribute)
	for i in range (0, len(tempDataTable)):
		if tempDataTable[i][index] <= value:
			leftDataTable.append(tempDataTable[i])
		else:
			rightDataTable.append(tempDataTable[i])
	return leftDataTable, rightDataTable

def homogenous(tempDataTable):
	pos, neg = getPosNeg(tempDataTable)
	if pos ==0 or neg == 0:
		return True
	else:
		return False

def overfitting(tempDataTable):
	if len(tempDataTable) < overfittingFactor:
		return True

def getOverfittingResult(tempDataTable):
	pos, neg = getPosNeg(tempDataTable)
	if pos>neg:
		return 'spam'
	else:
		return 'notspam'

#traversing the data tree to test the document as spam or notspam
def traverseTree(root, dataRow, attributes):
	temp = root
	while temp.result == None:
		tempAttr = temp.attribute
		tempFactor = temp.factor
		dataRowValue = 0
		if tempAttr in attributes:
			attrIndex = attributes.index(tempAttr)
			dataRowValue = dataRow[attrIndex]
		if dataRowValue <= tempFactor:
			temp = temp.left
		else:
			temp = temp.right
	return temp.result

#idea to print tree using level order traversal adapted from
#http://www.geeksforgeeks.org/level-order-tree-traversal/
def printDecisionTree(root):
    calculatedHeight = treeHeight(root)
    if calculatedHeight > 4:
        calculatedHeight = 5
    for i in range(1, calculatedHeight):
        print '--------------------'
        print 'Nodes at level %d' % (i)
        printLevel(root, i)

def printLevel(root, level):
    if root is None:
        return
    if level == 1:
        print 'Word: %s' % (root.attribute)
    elif level > 1:
        printLevel(root.left, level -1)
        printLevel(root.right, level -1)

def treeHeight(root):
    if root is None:
        return 0
    else:
        leftSubtreeHeight = treeHeight(root.left)
        rightSubtreeHeight = treeHeight(root.right)
        return (leftSubtreeHeight + 1) if (leftSubtreeHeight > rightSubtreeHeight) else (rightSubtreeHeight + 1)