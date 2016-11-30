import copy
class node:
    def __init__(self, attribute, dataTable):
        self.attribute = attribute  #the word on which splitting happens
        self.left = None    #left subtree that does not contain the attribute
        self.right = None   #right subtree that contains the attribute
        self.parent = None  #parent of this node
        self.result = None    #result at this node
        self.dataTable = dataTable  #datatable at this node
        self.positive = 0   #positive for spam
        self.negative = 0   #negative for notspam