class decisionTree:
    def __init__(self, attribute, factor, dataTable):
        self.attribute = attribute  #the word on which splitting happens
        self.factor = factor    #split factor
        self.left = None    #left subtree
        self.right = None   #right subtree
        self.parent = None  #parent of this node
        self.result = ''    #result at this node
        self.dataTable = dataTable  #datatable at this node

    def hasLeft(self):
        return self.left

    def hasRigth(self):
        return self.right

    def addLeft(self, leftNode):
        self.left = leftNode

    def addRight(self, rightNode):
        self.right = rightNode

    def setDataTable(self, dataTable):
        self.dataTable.dataTable

    def setParent(self, parent):
        self.parent = parent