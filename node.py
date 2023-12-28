class Node:

    def __init__(self, gini, gini_index):
        self.gini = gini # Gini value
        self.gini_index = gini_index # Vector, representing the gini index
        self.left = None
        self.right = None
