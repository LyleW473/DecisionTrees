import torch

# Data from lecture notes (Week 3 Intro to AI module)
# 0 = No | 1 = Yes 
data = [
        [1, 1],
        [0, 1],
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
        [1, 0],
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 1],
        [1, 0],
        [1, 1]
        ]
data = torch.tensor(data)

# Corresponding labels
# 0 = High, 1 = Low, 2 = Medium
labels = torch.tensor([1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2, 0, 1])

combined = torch.concat((data, labels.view(-1, 1)), dim = 1)
print(combined, combined.shape)

# Calculate gini-values
num_entries, num_features = data.shape
classes, classes_counts = torch.unique(labels, return_counts = True)
print("classes:", classes, classes_counts)

for feature_idx in range(0, num_features):
    feature_column = data[torch.arange(num_entries), torch.zeros(num_entries, dtype = torch.long) + feature_idx]
    print("feature_column", feature_column)
    possible_values, value_counts = torch.unique(feature_column, return_counts = True)
    total_elements_in_node = torch.sum(value_counts)
    print(possible_values, value_counts)
    


    classification_counts = []
    for classification_class in classes:
        print("c", classification_class)
        # Find the values in the feature column where e.g., class == High
        print(labels == classification_class)
        print("here", feature_column[labels == classification_class])
        matching_feature_vals_and_class = feature_column[labels == classification_class]

        # Count number of "Yes" or "No" for this feature column
        # (2, 4) = 2 "No"s mapped to "High", 4 "Yes"s mapped to "High", where "High" is the classification class
        counts = torch.tensor([torch.sum(matching_feature_vals_and_class == val) for val in possible_values])
        
        # Add the counts to the 
        classification_counts.append(counts)

    # Create matrix using the classification counts
    gini_matrix = torch.stack(classification_counts, dim = 0)
    gini_vectors = gini_matrix.T # Transpose the matrix to get the gini vectors, e.g., Gini(2, 0, 4), etc...
    print(gini_matrix)
    print(gini_vectors)
    print()
