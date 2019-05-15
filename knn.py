import pandas as pd
import numpy as np
from fastai.structured import  add_datepart
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import db_functions

scaler = MinMaxScaler(feature_range=(0, 1))

data = db_functions.getDb()
add_datepart(data, 'Date')
data.drop('Elapsed', axis=1, inplace=True)

train = data[:987]
valid = data[987:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']


x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

model.fit(x_train,y_train)
prediction = model.predict(x_valid)

print(db_functions.getRMSE(valid, prediction))

db_functions.plot(train, valid, prediction)
