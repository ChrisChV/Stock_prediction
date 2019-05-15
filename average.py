import db_functions

data = db_functions.getDb()

train = data[:987]
test = data[987:]

predictions = []

for i in range(0, len(test)):
    a = train['Close'][len(train)-len(test)+i:].sum() + sum(predictions)
    b = a/len(test)
    predictions.append(b)

db_functions.plot(train, test, predictions)
print db_functions.getRMSE(test, predictions)

