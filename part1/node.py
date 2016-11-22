class Node:
    def __init__(self, attribute, splitValue):
        self.attribute = attribute  #the attribute or word for this node
        self.splitValue = splitValue    #the split value to split between left and right subtree
        self.left = None
        self.right = None