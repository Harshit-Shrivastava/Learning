import copy
class node:
    def __init__(self, attribute, factor, dataTable):
        self.attribute = attribute  #the word on which splitting happens
        self.factor = factor    #split factor
        self.left = None    #left subtree
        self.right = None   #right subtree
        self.parent = None  #parent of this node
        self.result = None    #result at this node
        self.dataTable = dataTable  #datatable at this node
        self.positive = 0   #positive for spam
        self.negative = 0   #negative for notspam

    def createNode(self, attribute, factor, dataTable):
        self.attribute = attribute  # the word on which splitting happens
        self.factor = factor  # split factor
        self.dataTable = copy.deepcopy(dataTable)  # datatable at this node