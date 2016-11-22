class Node:
    def __init__(self, attribute, splitValue):
        self.attribute = attribute  #word to check for at this node
        self.splitValue = splitValue    #the split value to split between left and right subtree
        self.left = None    #left child
        self.right = None   #right child
        self.result = ''    #indicates a decision between spam and non-spam