import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import sklearn

iris = sklearn.datasets.load_iris()
X, y = iris.data, iris.target


class MLP(nn.Module):
    def __init__(self, cols, size_hidden, classes):
        super(MLP, self).__init__()
        # Note that 17 is the number of columns in the input matrix.
        self.fc1 = nn.Linear(cols, size_hidden)
        # variety of # possible for hidden layer size is arbitrary, but needs to be consistent across layers.  3 is the number of classes in the output (died/survived)
        self.fc2 = nn.Linear(size_hidden, classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class ResilientMLP(nn.Module):
    def __init__(self, cols, size_hidden, classes):
        super(ResilientMLP, self).__init__()
        # Note that 17 is the number of columns in the input matrix.
        self.fc1 = nn.Linear(cols, size_hidden)
        # variety of # possible for hidden layer size is arbitrary, but needs to be consistent across layers.  3 is the number of classes in the output (died/survived)
        self.fc2 = nn.Linear(size_hidden, classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class ResilientLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, res_coeff: int = 0, bias: bool = True,
                 device=None, dtype=None) -> None:
        super(ResilientLinear, self).__init__(in_features - 2 * res_coeff, out_features, bias, device, dtype)
        self.f = res_coeff

    def forward(self, input: Tensor) -> Tensor:
        rankings = np.argsort(input.tolist())
        removes = np.concatenate((rankings[:self.f], rankings[-self.f:]))
        
