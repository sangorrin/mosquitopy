import pandas as pd
import numpy as np

train_data = pd.read_csv('./train.csv')

def ll(text):
    return int(float(text)*10000)

train_data['NormLatitude'] = train_data['Latitude'].apply(ll)
train_data['NormLongitude'] = train_data['Longitude'].apply(ll)

X = train_data[['NormLatitude', 'NormLongitude']]
y = train_data['WnvPresent']


X_train = X[:len(X)/2]
y_train = y[:len(X)/2]
X_valid = X[len(X)/2:]
y_valid = y[len(X)/2:]

from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)
