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
from sklearn.ensemble import RandomForestRegressor

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

	is_sj = data.city == 'sj'

	months = data.week_start_date.apply(extract_month)

	data.drop(['city', 'weekofyear', 'week_start_date'], axis=1, inplace=True)

	scaler = StandardScaler()

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

V = preprocess(V, False)


# 0 = linear regression
# 1 = random forrest

algorithm = 1

if algorithm == 0:
	# Linear Regression
	lr = LinearRegression()
	lr.fit(X_train, Y_train)
	Y_pred = lr.predict(X_test)
	# print(mean_absolute_error(Y_test, Y_pred))

	predictions = lr.predict(V)
	predictions[predictions < 0] = 0

elif algorithm == 1:
	# Random forrest classifier
	rf = RandomForestRegressor(n_estimators = 300)

	rf.fit(X_train, Y_train);
	Y_pred = rf.predict(X_test)
	print(mean_absolute_error(Y_test, Y_pred))

	predictions = rf.predict(V)


predictions = predictions.astype(int)
print(predictions)

# Save predictions to file
with open('predict', 'w') as csvfile2:
    writer=csv.writer(csvfile2, delimiter=',')
    writer.writerows(zip(predictions))
    print('klaar')

exit()



