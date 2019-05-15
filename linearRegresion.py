import db_functions
from fastai.structured import  add_datepart
from sklearn.linear_model import LinearRegression

data = db_functions.getDb()
add_datepart(data, 'Date')
data.drop('Elapsed', axis=1, inplace=True)


#data['mon_fri'] = 0
#for i in range(0,len(data)):
#    if (data['Dayofweek'][i] == 0 or data['Dayofweek'][i] == 4):
#        data['mon_fri'][i] = 1
#   else:
#        data['mon_fri'][i] = 0
    
train = data[:987]
valid = data[987:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

model = LinearRegression()
model.fit(x_train,y_train)

prediction = model.predict(x_valid)

db_functions.plot(train, valid, prediction)
print(db_functions.getRMSE(valid, prediction))