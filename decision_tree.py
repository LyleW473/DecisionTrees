import torch
from node import Node
class DecisionTree:

    def __init__(self, data, labels):

        self.classes = torch.unique(labels)
        self.num_classes = self.classes.shape[0]

        # Calculate the gini index and gini value for the first node
        _, gini_index = torch.unique(labels, return_counts = True)
        divide_counts = gini_index / torch.sum(gini_index, dim = 0)
        probabilities = divide_counts * (1 - divide_counts)
        gini_value = torch.sum(probabilities, dim = 0)

        # Construct tree
        self.tree = Node(gini = gini_value, gini_index = gini_index)
        self.tree = self._construct_tree(data, labels, self.tree)
        
    def _construct_tree(self, data, labels, node):

        # Have not reached a leaf node
        if node.gini > 0:

            # Find the best feature to split on from this point in the tree
            gini_values_hashmap, feature_gini_vectors_hashmap = self._find_gini_values(data = data, labels = labels)
            minimum_feature_index = self._find_minimum_gini_split(
                                        num_features = data.shape[1], 
                                        num_entries = data.shape[0], 
                                        feature_gini_vectors_hashmap = feature_gini_vectors_hashmap,
                                        gini_values_hashmap = gini_values_hashmap
                                        )
            
            # print(gini_values_hashmap, data.shape)
            print("Min index", minimum_feature_index)
            # Entries inside of the feature column that are 0/1 (e.g., OwnsHome <= 0.5)
            data_0 = (data[torch.arange(data.shape[0]), minimum_feature_index] == 0)
            data_1 = (data[torch.arange(data.shape[0]), minimum_feature_index] == 1)

            gini_vectors = feature_gini_vectors_hashmap[minimum_feature_index.item()]
            # True
            node.left = self._construct_tree(
                                            data = data[data_0].clone(), 
                                            labels = labels[data_0].clone(), 
                                            node = Node(gini = gini_values_hashmap[gini_vectors[0]], gini_index = gini_vectors[0])
                                            )
            # False
            node.right = self._construct_tree(
                                            data = data[data_1].clone(), 
                                            labels = labels[data_1].clone(), 
                                            node = Node(gini = gini_values_hashmap[gini_vectors[1]], gini_index = gini_vectors[1])
                                            )

        print("return:", node.gini_index, node.gini, node.left.gini if node.left else "No Left", node.right.gini if node.right else "No Right")

        return node

    def _find_gini_values(self, data, labels):
        # Calculate gini-values
        num_entries, num_features = data.shape

        gini_values_hashmap = {} # Maps gini vectors to their corresponding gini values
        feature_gini_vectors_hashmap = {i:[] for i in range(0, num_features)} # Maps each feature to their gini vectors

        for feature_idx in range(0, num_features):
            # Extract the feature column from the "data" matrix
            feature_column = data[torch.arange(num_entries), torch.zeros(num_entries, dtype = torch.long) + feature_idx]
            # print("feature_column", feature_column)
            
            # Find the possible values that can be inside the column (e.g., "Yes" and "No") and the number of occurrences
            possible_values, value_counts = torch.unique(feature_column, return_counts = True)
            # print(possible_values, value_counts)

            classification_counts = []
            for classification_class in self.classes:
                # print("c", classification_class)
                # Find the values in the feature column where e.g., class == High
                # print(labels == classification_class)
                # print("here", feature_column[labels == classification_class])
                matching_feature_vals_and_class = feature_column[labels == classification_class]

                # Count number of "Yes" or "No" for this feature column
                # (2, 4) = 2 "No"s mapped to "High", 4 "Yes"s mapped to "High", where "High" is the classification class
                counts = torch.tensor([torch.sum(matching_feature_vals_and_class == val) for val in possible_values])
                
                # Add the counts to the 
                classification_counts.append(counts)

            # Create matrix using the classification counts
            gini_matrix = torch.stack(classification_counts, dim = 0)
            gini_vectors = gini_matrix.T # Transpose the matrix to get the gini vectors, e.g., Gini(2, 0, 4), etc...
            # print(gini_matrix)
            # print(gini_vectors)
            
            # Finding gini values for each vector
            gini_vector_sums = gini_vectors.sum(dim = 1, keepdim = True) # Find sums of each row vector
            divided_gini_vectors = gini_vectors / gini_vector_sums # Convert 2 and 4 into (2/6) and (4/6), respectively
            final_gini_vectors = divided_gini_vectors * (1 - divided_gini_vectors) # (2/6) * (1 - (2/6) For each item in the gini vector

            # print(gini_vector_sums)
            # print(divided_gini_vectors)
            # print(final_gini_vectors)

            # Calculate and save the gini values for each gini vector inside the hashmap
            for gini_vector, final_gini_vector in zip(gini_vectors, final_gini_vectors):
                gini_value = torch.sum(final_gini_vector, dim = 0) # Summation to find gini value
                gini_values_hashmap[gini_vector] = gini_value
                feature_gini_vectors_hashmap[feature_idx].append(gini_vector)
                # print(gini_vector, gini_value)

        # print("Gini values hashmap:", gini_values_hashmap)
        return gini_values_hashmap, feature_gini_vectors_hashmap

    def _find_minimum_gini_split(self, num_features, num_entries, feature_gini_vectors_hashmap, gini_values_hashmap):

        # Calculating Gini-Split for each feature
        gini_splits = []
        for feature_idx in range(0, num_features):
            gini_vectors_for_feature = feature_gini_vectors_hashmap[feature_idx]
            probabilities = torch.tensor([torch.sum(gini_vector) / num_entries for gini_vector in gini_vectors_for_feature])
            gini_values = torch.tensor([gini_values_hashmap[gini_vector] for gini_vector in gini_vectors_for_feature])
            # print(gini_vectors_for_feature)
            # print(probabilities)
            # print(gini_values)
            # print(probabilities * gini_values)
            gini_split = torch.sum(probabilities * gini_values, dim = 0)
            gini_splits.append(gini_split)
            # print(f"Gini split for feature {feature_idx}: {gini_split}")
        
        return torch.argmin(torch.tensor(gini_splits)) # Returns feature index of the feature with the minimum gini split