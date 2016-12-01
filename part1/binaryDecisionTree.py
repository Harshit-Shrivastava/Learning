import operator
from decisionTree import getPosNeg
from decisionTree import homogenous
from decisionTree import overfitting
from decisionTree import getOverfittingResult
from decisionTree import getEntropy
from binaryNode import node

def binaryDecisionTree(tempDataTable, attributes, wordList):
    if len(wordList) == 0:
        pos, neg = getPosNeg(tempDataTable)
        root = node(None, None)
        if pos > neg:
            root.result = 'spam'
        else:
            root.result = 'notspam'
        return root
    if (homogenous(tempDataTable)):
        pos, neg = getPosNeg(tempDataTable)
        root = node(None, None)
        if pos > neg:
            root.result = 'spam'
        else:
            root.result = 'notspam'
        return root
    if (overfitting(tempDataTable)):
        root = node(None, None)
        root.result = getOverfittingResult(tempDataTable)
        return root
    attribute= selectAttribute(attributes, wordList, tempDataTable)
    # rows that do not have the attribute form the left sub-tree,
    # rows that have the attribute form the right sub-tree
    leftDataTable, rightDataTable = splitDataTable(attribute, attributes, tempDataTable)
    pos, neg = getPosNeg(tempDataTable)  # count of spams and notspams
    root = node(attribute, tempDataTable)  # create the node with reqd info
    root.positive = pos  # positive for spam
    root.negative = neg  # negative for not spam
    wordList.remove(attribute)
    root.left = binaryDecisionTree(leftDataTable, attributes, wordList)  # recursively create left subtree
    root.right = binaryDecisionTree(rightDataTable, attributes, wordList)  # recursively create right subtree
    return root

def selectAttribute(attributes, wordList, dataTable):
    gains = []
    words = []
    for w in attributes:
        if w not in wordList:
            continue
        else:
            gains.append(findGain(w, attributes, dataTable))
            words.append(w)
    max_index, max_value = max(enumerate(gains), key=operator.itemgetter(1))
    max_gain_word = words[max_index]
    word_index = attributes.index(max_gain_word)
    return attributes[word_index]

def findGain(attribute, attributes, dataTable):
	wordIndex = attributes.index(attribute)
	wholeColumn = [ row[wordIndex] for row in dataTable]
	less = []
	more = []
	for i in range(len(wholeColumn)):
		if wholeColumn[i] == 0:
			less.append(dataTable[i])
		else:
			more.append(dataTable[i])
	e,e1,e2 = getEntropy(dataTable),getEntropy(less),getEntropy(more)
	w1 = float(len(less))/len(dataTable)
	w2 = float(len(more))/len(dataTable)
	gain = e - (e1*w1 + e2*w2)
	return gain

def splitDataTable(attribute, attributes, tempDataTable):
    leftDataTable = []
    rightDataTable = []
    index = attributes.index(attribute)
    # rows (documents/spams) having value 0 for this attribute go to the left, others to the right
    for i in range(0, len(tempDataTable)):
        if tempDataTable[i][index] == 0:
            leftDataTable.append(tempDataTable[i])
        else:
            rightDataTable.append(tempDataTable[i])
    return leftDataTable, rightDataTable

#traversing the data tree to test the document as spam or notspam
def traverseBinaryTree(root, dataRow, attributes):
	temp = root
	while temp.result == None:
		tempAttr = temp.attribute
		dataRowValue = 0
		if tempAttr in attributes:
			attrIndex = attributes.index(tempAttr)
			dataRowValue = dataRow[attrIndex]
		if dataRowValue == 0:
			temp = temp.left
		else:
			temp = temp.right
	return temp.result

#idea to print tree using level order traversal adapted from
#http://www.geeksforgeeks.org/level-order-tree-traversal/
def printBinaryDecisionTree(root):
    calculatedHeight = binaryTreeHeight(root)
    if calculatedHeight > 4:
        calculatedHeight = 5
    print '--------------------'
    for i in range(1, calculatedHeight):
        print 'Nodes at level %d' %(i)
        printBinaryLevel(root, i)
        print '--------------------'

def printBinaryLevel(root, level):
    if root is None:
        return
    if level == 1:
        print 'Word: %s' % (root.attribute)
    elif level > 1:
        printBinaryLevel(root.left, level -1)
        printBinaryLevel(root.right, level -1)

def binaryTreeHeight(root):
    if root is None:
        return 0
    else:
        leftSubtreeHeight = binaryTreeHeight(root.left)
        rightSubtreeHeight = binaryTreeHeight(root.right)
        return (leftSubtreeHeight + 1) if (leftSubtreeHeight > rightSubtreeHeight) else (rightSubtreeHeight + 1)