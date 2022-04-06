import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
        self.fc2 = ResilientLinear(size_hidden, classes)

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
        masks = torch.sparse_coo_tensor([removes.tolist()], np.ones(removes.shape).tolist(), input.shape)
        filtered_input = input[masks != 1]
        return F.linear(filtered_input, self.weight, self.bias)


iris = load_iris()
X, y = iris.data, iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
EPOCHS = 1000

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)
X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test = Variable(torch.from_numpy(X_test)).float()
y_test = Variable(torch.from_numpy(y_test)).long()

model_plain = MLP(X.shape[1], 50, 3)
optimizer = torch.optim.Adam(model_plain.parameters(), lr = 0.001)
loss_fn = nn.CrossEntropyLoss()

loss_list = np.zeros((EPOCHS,))
accuracy_list = np.zeros((EPOCHS,))

import tqdm

for epoch in tqdm.trange(EPOCHS):
    y_pred = model_plain(X_train)
    loss = loss_fn(y_pred, y_train)
    loss_list[epoch] = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        y_pred = model_plain(X_test)
        correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list[epoch] = correct.mean()


import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

ax1.plot(accuracy_list)
ax1.set_ylabel("validation accuracy")
ax2.plot(loss_list)
ax2.set_ylabel("validation loss")
ax2.set_xlabel("epochs")

fig.savefig("plain.pdf")
