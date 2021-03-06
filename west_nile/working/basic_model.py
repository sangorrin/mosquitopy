import pandas as pd
import numpy as np

train_data = pd.read_csv('../input/train.csv')

def ll(text):
    return int(float(text)*10000)
train_data['NormLatitude'] = train_data['Latitude'].apply(ll)
train_data['NormLongitude'] = train_data['Longitude'].apply(ll)

X = train_data[['NormLatitude', 'NormLongitude', 'Species']]
y = train_data['WnvPresent']


X_train = X[:len(X)/2]
y_train = y[:len(X)/2]

X_valid = X[len(X)/2:]
y_valid = y[len(X)/2:]


from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve

def report_score(model, X_valid, y_valid):
    y_predict = model.predict(X_valid)
    print 'Recall: ', recall_score(y_valid, y_predict)
    print 'Acc: ', accuracy_score(y_valid, y_predict)
    print 'F1:', f1_score(y_valid, y_predict)

from sklearn.neighbors import KNeighborsClassifier
for k in range(1, 6):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    print '\n\nKNN k=%d' % k
    report_score(knn, X_valid, y_valid)

