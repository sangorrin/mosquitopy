import pandas as pd
import numpy as np

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve


train_data = pd.read_csv('../input/train.csv')


def ll(text):
    return str(float(text)*10000)

def year(date_text):
    return date_text[:6]

train_data['NormLatitude'] = train_data['Latitude'].apply(ll)
train_data['NormLongitude'] = train_data['Longitude'].apply(ll)
train_data['Year'] = train_data['Date'].apply(year)


X = train_data[['Year', 'Species']]
y = train_data['WnvPresent']



from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X = vec.fit_transform(X.T.to_dict().values())

# Shuffling data
shuffle_index = np.arange(len(X))
np.random.shuffle(shuffle_index)
X = X[shuffle_index]
y = y[shuffle_index]

# Split data
valid_size = len(X)/4
X_train = X[:valid_size]
y_train = y[:valid_size]
X_valid = X[valid_size:]
y_valid = y[valid_size:]

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



