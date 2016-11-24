import copy
import numpy as np
from node import node

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

#calculates the splitting factor.
#this needs to be decided, for now, mean is taken as the splitting factor
def getSplitFactor(attribute, tempDataTable):
	x, y = getIndex(attribute, tempDataTable)
	temp = []
	for i in range(1, len(tempDataTable)):
		temp.append(tempDataTable[i][y])
	return np.array(temp).mean(0)

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
	for i in range (1,len(tempDataTable)):
		if tempDataTable[i][len(tempDataTable[0]) - 1] == 1:
			pos += 1
		else:
			neg += 1
	return pos, neg

#this function creates two subtables with values for the attribute less than and greater than
#the splitting factor. It later removes the column for this splitting factor from the two subtables
#as the splitting on this factor has already happened
def splitTrainingset(attribute, value, tempDataTable):
	leftDataTable = copy.deepcopy(tempDataTable)
	rightDataTable = copy.deepcopy(tempDataTable)
	x, y = getIndex(attribute, tempDataTable)
	for i in range (1, len(tempDataTable)):
		if tempDataTable[i][y] > value:
			leftDataTable.remove(tempDataTable[i])
		else:
			rightDataTable.remove(tempDataTable[i])
	#leftDataTable = transpose(leftDataTable)
	#code from http://stackoverflow.com/questions/42519/how-do-you-rotate-a-two-dimensional-array
	#leftDataTable = zip(*leftDataTable)[::-1]
	#rightDataTable = transpose(rightDataTable)
	#rightDataTable = zip(*rightDataTable)[::-1]
	#leftDataTable.remove(leftDataTable[y])
	#rightDataTable.remove(rightDataTable[y])
	return leftDataTable, rightDataTable

def transpose(grid):
	return zip(*grid)