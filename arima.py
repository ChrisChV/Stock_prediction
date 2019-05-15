from pyramid.arima import auto_arima
import pandas as pd
import db_functions

data = db_functions.getDb()

data = data.sort_index(ascending=True, axis=0)

train = data[:987]
valid = data[987:]

training = train['Close']
validation = valid['Close']

model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model.fit(training)
forecast = model.predict(n_periods=248)



print(db_functions.getRMSE(valid, forecast))
db_functions.plot(train, valid, forecast)
