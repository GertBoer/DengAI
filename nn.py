import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

import csv
import math
import time
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


X = pd.read_csv("data/dengue_features_train.csv")
y = pd.read_csv("data/dengue_labels_train.csv")
V = pd.read_csv("data/dengue_features_test.csv")


def extract_month(s):
    return int(s[5:7])

def preprocess(data, train):
	if train == True:
		data.dropna(inplace=True)
		data.fillna(0,inplace=True)
	else:
		data.fillna(0,inplace=True)

	is_sj = data.city == 'sj'

	months = data.week_start_date.apply(extract_month)

	data.drop(['city', 'weekofyear', 'week_start_date'], axis=1, inplace=True)

	scaler = MinMaxScaler()

	data[data.columns] = scaler.fit_transform(data)

	data['is_sj'] = is_sj.loc[data.index]

	sliced_months = months.loc[data.index]

	month_features = pd.get_dummies(sliced_months)
	data = data.join(sliced_months)

	return data

X = preprocess(X, True)

y = y[['total_cases']]
y = y.total_cases.loc[X.index]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, shuffle = False)



# V = preprocess(V, False)

X_train = X_train.astype(float)
Y_train = Y_train.astype(float)

X_train = Variable(torch.tensor(X_train.values)).float()
Y_train = Variable(torch.tensor(Y_train.values)).float()





epochs = 10
learning_rate = 0.001


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        l1 = self.linear1(x)
        l2 = self.linear2(l1)
        y_pred = self.linear3(l2)
        return y_pred




# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 23, 300, 1

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for t in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(X_train)

    # Compute and print loss
    loss = criterion(y_pred, Y_train)
    # print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t == 998:
    	print(y_pred)











# def train_model:
# 	pass


# def test_model: 
# 	pass


# for epoch in range(epochs):
# 	pass 









