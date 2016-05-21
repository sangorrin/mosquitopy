import pandas as pd
import numpy as np

train_data = pd.read_csv('../input/train.csv')

def ll(text):
    return int(float(text)*10000)

X = train_data[['Latitude', 'Longitude']]
y = train_data['WnvPresent']


X_train = X[:len(X)/2]
y_train = y[:len(X)/2]

X_valid = X[len(X)/2:]
y_valid = y[len(X)/2:]


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


print knn.score(X_valid, y_valid)