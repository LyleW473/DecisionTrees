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
print(classes, classes_counts)

for feature_idx in range(0, num_features):
    column_values = data[torch.arange(num_entries), torch.zeros(num_entries, dtype = torch.long) + feature_idx]
    print(column_values)
    unique_values, value_counts = torch.unique(column_values, return_counts = True)
    total_elements_in_node = torch.sum(value_counts)
    print(unique_values, value_counts)
