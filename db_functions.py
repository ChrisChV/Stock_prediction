import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

DATABASE_FILE_NAME = "NSE-TATAGLOBAL11.csv"


def getDb():
    rcParams['figure.figsize'] = 20,10
    df = pd.read_csv(DATABASE_FILE_NAME)

    df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
    df.index = df['Date']

    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]

    return new_data

def getRMSE(valid, prediction):
    rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-prediction),2)))
    return rms

def plot(train, valid, prediction):
    valid['Predictions'] = 0
    valid['Predictions'] = prediction
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.show()

