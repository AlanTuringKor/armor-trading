import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import yfinance as yf
import tensorflow as tf
import requests

from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score

# TODO: need saving mechanism


def _get_hashrate(start_date, end_date):
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    end_date = end_date + timedelta(days=1)
    params = {
    'assets': 'btc',
    'page_size': 10000,
    'metrics': 'HashRate',
    'start_time': start_date.strftime("%Y-%m-%d"),
    'end_time': end_date.strftime("%Y-%m-%d"),  # or a specific end date
    }

    response = requests.get(url, params=params)
    data = response.json()['data']
    data = [item['HashRate'] for item in data]

    return data
    
def _get_transaction_count(start_date, end_date):
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    end_date = end_date + timedelta(days=1)
    params = {
    'assets': 'btc',
    'page_size': 10000,
    'metrics': 'TxCnt',
    'start_time': start_date.strftime("%Y-%m-%d"),
    'end_time': end_date.strftime("%Y-%m-%d"),  # or a specific end date
    }

    response = requests.get(url, params=params)
    data = response.json()['data']
    data = [item['TxCnt'] for item in data]

    return data
        


# type = 1: only Price, type = 2: Price and Volume, type = 3: Price, Volume, Transaction data and Hash rate
def get_data(type, start_date, end_date):
    # get Price and Volume

    

    df = yf.download(f'BTC-USD', start=start_date, end=end_date)
    
    print("the most recent date of the data is:" + df.tail(1).index[0].strftime('%Y-%m-%d %H:%M:%S'))

    # Printing the first few entries of the dataset
    print("\n\nThe head of the dataframe of {}: \n\n".format("BTC"))
    print(df.head())

    print("\n\n Description of the dataset of {}: \n\n".format("BTC"))
    print(df.describe())

    # Pre-Processing

    # Scaling the data using the Min Max Scaling between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    #df = df[['Close', 'Volume']].values.reshape(-1, 2)
    close = df[['Close']].values
    volume = df[['Volume']].values
    hashrate = np.array(_get_hashrate(start_date, end_date)).reshape(-1, 1)
    txCount = np.array(_get_transaction_count(start_date, end_date)).reshape(-1, 1)
    
    # if the length of those 4 numpy array is not same than delete the latest Element of each list so the length of those list corresponds the length of the shortest list 
    # Find the minimum length among all arrays
    min_length = min(len(close), len(volume), len(hashrate), len(txCount))

    # Truncate each array to the minimum length
    close = close[:min_length]
    volume = volume[:min_length]
    hashrate = hashrate[:min_length]
    txCount = txCount[:min_length]
    
    df = np.concatenate((close,volume,hashrate,txCount),axis=1)

    df_scaled = scaler.fit_transform(df.reshape(-1, 4))
    
    x_train, y_train = [], []
    # Looks back on 60 days of data to predict the values of 61st day
    lookback = 60
    # Filling up the x_train and y_train with the scaled data
    for i in range(lookback, len(df_scaled)):

        x_train.append(df_scaled[i - lookback: i, 0:4])

        # The value of Closing price at i is the the required output/label
        y_train.append(df_scaled[i, 0])


    # Converting the data set we have created into a numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    
    print("\n\n The number of samples in our training data = " + str(len(x_train)) + "\n\n")

    # get Transaction data and Hash rate
    raise NotImplementedError()

def _create_and_fit_model_LSTM(layers, train, target):
    model = Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(LSTM(units=layers[i], return_sequences=True, input_shape=(train.shape[1], train.shape[2])))
            model.add(Dropout(0.2))
        elif i == len(layers)-1:
            model.add(LSTM(units=layers[i]))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
        else:
            model.add(LSTM(units=layers[i], return_sequences=True))
            model.add(Dropout(0.2))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    history = model.fit(train, target, epochs=25, batch_size=32)

    return model


def _visualize_training_history(history):
    historyForPlot = pd.DataFrame(history.history)
    historyForPlot.index += 1 # we plus 1 to the number of indexing so our epochs Plot picture will be counting from 1 not 0.
    historyForPlot.plot()
    plt.ylabel("loss")
    plt.xlabel("epochs")

def _create_and_fit_model_BILSTM(layers, train, target):
    raise NotImplementedError()

def _create_and_fit_model_GRU(layers, train, target):
    raise NotImplementedError()

def create_and_fit_model(type, layers, train, target):
    """
    This is the public function for creating and fitting a Model.

    Args:
        type (str): The type of the desired Model.("LSTM", "GRU", "BILSTM")
        layers (list of int): The List contains number of the Units which the Hidden Layers gonna have. Should have at least 2 Elements
        train (array): Training data
        target (array): Target data

    Returns:
        Model: the created and fitted model
    """
    if type == "LSTM":
        return _create_and_fit_model_LSTM(layers, train, target)
    elif type == "GRU":
        return _create_and_fit_model_GRU(layers, train, target)
    elif type == "BILSTM":
        return _create_and_fit_model_BILSTM(layers, train, target)
    else:
        print("wrong input bro! only one of these (LSTM, GRU, BILSTM)")

    raise NotImplementedError()

def test_and_visualize_model(model, regression_steps):
    raise NotImplementedError()






#################################################

start_date = dt.datetime(2021, 1, 1)
end_date = dt.datetime(2023, 12, 24)
#end_date = dt.datetime.now()

get_data(1,start_date,  end_date)


