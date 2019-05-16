from fbprophet import Prophet
import db_functions


data = db_functions.getDb()

data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)


train = data[:987]
valid = data[987:]

model = Prophet()
model.fit(train)

close_prices = model.make_future_dataframe(periods=len(valid))
forecast = model.predict(close_prices)

forecast_valid = forecast['yhat'][987:]
valid.rename(columns={'y': 'Close'}, inplace=True)
train.rename(columns={'y': 'Close'}, inplace=True)

print(db_functions.getRMSE(valid, forecast_valid))

db_functions.plot(train, valid, forecast_valid)