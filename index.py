import torch
import torch.nn as nn
import pandas as pd
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import csv
import math

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

# used as example: https://github.com/SusmeetJain/dengue_prediction/blob/master/1.Simple_Linear_Models.ipynb

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


	months = data.week_start_date.apply(extract_month)

	data.drop(['city', 'year', 'weekofyear', 'week_start_date'], axis=1, inplace=True)

	scaler = MinMaxScaler()

	data[data.columns] = scaler.fit_transform(data)

	sliced_months = months.loc[data.index]

	month_features = pd.get_dummies(sliced_months)
	data = data.join(sliced_months)

	return data

X = preprocess(X, True)
y = y[['total_cases']]
y = y.total_cases.loc[X.index]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, shuffle = False)


lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
print(mean_absolute_error(Y_test, Y_pred))

V = preprocess(V, False)

print(V)
predictions = lr.predict(V)

# predictions = np.sqrt(predictions**2)
# predictions = predictions.astype(int)

print(predictions)

with open('predict', 'w') as csvfile2:
    writer=csv.writer(csvfile2, delimiter=',')
    writer.writerows(zip(predictions))
    print('klaar')

exit()





# Experimented with a Pytorch.

# y_p = np.full(len(Y_test), np.mean(Y_train))
# print(mean_absolute_error(Y_test, y_p))

class LinearRegressionModel(nn.Module):

	def __init__(self, input_dim, output_dim):

		super(LinearRegressionModel, self).__init__() # its just like c++ super in inheritance bringing imp things to out class
		self.linear = nn.Linear(input_dim, output_dim) # nn.linear is defined in nn.Module

	def forward(self, x):# here its simple as we want our model to predict the output as what it thinks is correct

		out = self.linear(x)

		return out


def train_model(model, inputs, labels, optimiser, criterion):
	optimiser.zero_grad()

	outputs = model.forward(inputs)
	loss = criterion(outputs, labels)
	loss.backward()# back props
	optimiser.step()# update the parameters
	print('epoch {}, loss {}'.format(epoch,loss.item()))


def test_model(model, test_input):
	print(model.forward(Variable(test_input)).data.numpy())



input_dim = 20
output_dim = 1

model = LinearRegressionModel(input_dim,output_dim)

criterion = nn.MSELoss()# Mean Squared Loss
l_rate = 0.001
optimiser = torch.optim.SGD(model.parameters(), lr = l_rate) 

epochs = 2000


X_train = Variable(torch.tensor(X_train)).float()
Y_train = Variable(torch.tensor(Y_train)).float()



for epoch in range(epochs):

	train_model(model, X_train, Y_train, optimiser, criterion)

	if epoch == 1 or epoch == 1500:
		test_model(model, test_input)

	

