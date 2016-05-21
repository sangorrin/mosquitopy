import pandas as pd
import numpy as np

from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve



train_data = pd.read_csv('../input/train.csv')
spray_data = pd.read_csv('../input/spray.csv')

print train_data['Latitude'].min(), train_data['Latitude'].max()
print train_data['Longitude'].min(), train_data['Longitude'].max()

spray_data['Latitude'].apply(float)
spray_data['Longitude'].apply(float)
spray_data = spray_data[ (41.60 < spray_data['Latitude']) & (spray_data['Latitude'] < 42.02)]
spray_data = spray_data[ (-87.94 < spray_data['Longitude']) & (spray_data['Longitude'] < -87.53)]

def ll(text):
    return str(int(float(text)*100))
train_data['NormLatitude'] = train_data['Latitude'].apply(ll)
train_data['NormLongitude'] = train_data['Longitude'].apply(ll)
spray_data['NormLatitude'] = spray_data['Latitude'].apply(ll)
spray_data['NormLongitude'] = spray_data['Longitude'].apply(ll)
train_data['IsSprayed'] = 0

import datetime
for spray_dict in spray_data.T.to_dict().values():
    date = spray_dict['Date']
    date = datetime.datetime.strptime(spray_dict['Date'], '%Y-%m-%d').date()

    same_location = \
        (train_data['NormLatitude'] == spray_dict['NormLatitude']) & \
         (train_data['NormLongitude'] == spray_dict['NormLongitude'])

    for i in range(0, 7):
        date = date + datetime.timedelta(days=1)
        same_date = (train_data['Date'] == date.isoformat())

        # set train_data['IsSprayed'] on the same_date and same_location = 1
