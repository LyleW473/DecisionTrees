import torch

# Data from lecture notes (Week 3 Intro to AI module)
# 0 = No | 1 = Yes 
data = torch.tensor(
    [
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
)

# Corresponding labels
# 0 = High, 1 = Low, 2 = Medium
labels = torch.tensor([1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2, 0, 1])