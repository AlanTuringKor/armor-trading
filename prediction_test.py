import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import yfinance as yf
import tensorflow as tf
import requests

from model import MyModel
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score





# type = 1: only Price, type = 2: Price and Volume, type = 3: Price, Volume, Transaction data and Hash rate
def get_data(type, start_date, end_date):
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
    
    x_data, y_data = [], []
    # Looks back on 60 days of data to predict the values of 61st day
    lookback = 60
    # Filling up the x_data and y_data with the scaled data
    for i in range(lookback, len(df_scaled)):

        x_data.append(df_scaled[i - lookback: i, 0:4])

        # The value of Closing price at i is the the required output/label
        y_data.append(df_scaled[i, 0])


    # Converting the data set we have created into a numpy array
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    
    print("\n\n The number of samples in our training data = " + str(len(x_data)) + "\n\n")
    
    return x_data, y_data

def _create_and_fit_model_LSTM(layers, x_data, y_data):
    model = Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(LSTM(units=layers[i], return_sequences=True, input_shape=(x_data.shape[1], x_data.shape[2])))
            model.add(Dropout(0.2))
        elif i == len(layers)-1:
            model.add(LSTM(units=layers[i]))
            model.add(Dropout(0.2))
            model.add(Dense(units=y_data.shape[2]))
        else:
            model.add(LSTM(units=layers[i], return_sequences=True))
            model.add(Dropout(0.2))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')
    # is the use of callback alright? is this like an Observer class?
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    history = model.fit(x_data, y_data, epochs=25, batch_size=32)

    return model, history


def _create_and_fit_model_BILSTM(layers, x_data, y_data):
    model = Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(Bidirectional(LSTM(units=layers[i], return_sequences=True), input_shape=(x_data.shape[1], x_data.shape[2])))
            model.add(Dropout(0.2))
        elif i == len(layers)-1:
            model.add(Bidirectional(LSTM(units=layers[i])))
            model.add(Dropout(0.2))
            model.add(Dense(units=y_data.shape[2]))
        else:
            model.add(Bidirectional(LSTM(units=layers[i], return_sequences=True)))
            model.add(Dropout(0.2))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')
    # is the use of callback alright? is this like an Observer class?
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    history = model.fit(x_data, y_data, epochs=25, batch_size=32)

    return model, history

def _create_and_fit_model_GRU(layers, x_data, y_data):
    model = Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(GRU(units=layers[i], return_sequences=True, input_shape=(x_data.shape[1], x_data.shape[2])))
            model.add(Dropout(0.2))
        elif i == len(layers)-1:
            model.add(GRU(units=layers[i]))
            model.add(Dropout(0.2))
            model.add(Dense(units=y_data.shape[2]))
        else:
            model.add(GRU(units=layers[i], return_sequences=True))
            model.add(Dropout(0.2))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')
    # is the use of callback alright? is this like an Observer class?
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    history = model.fit(x_data, y_data, epochs=25, batch_size=32)

    return model, history

def _visualize_training_history(history):
    historyForPlot = pd.DataFrame(history.history)
    historyForPlot.index += 1 # we plus 1 to the number of indexing so our epochs Plot picture will be counting from 1 not 0.
    historyForPlot.plot()
    plt.ylabel("loss")
    plt.xlabel("epochs")


# this function also should be transfers to model class. only prediction and visualization are suiting for this script
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

def test_and_visualize_model(model, data, start_date, regression_steps):
    # Create time values for the x-axis
    num_data_points = len(data)  # Assuming actual_prices and prediction_prices have the same length
    time_values = [start_date + datetime.timedelta(days=i) for i in range(num_data_points)]
    
    ####### varition 1 :test with regression_steps of 1 #######
    prediction_closing_prices = model.predict(data)
    prediction_closing_prices = scaler.inverse_transform(prediction_closing_prices)
    actual_closing_prices = data[:,0,0] 
     
    # varition 1-1: comparing graphs itself
    # Plot the data with time values on the x-axis
    plt.plot(time_values, actual_closing_prices, color='black', label='Actual Prices')
    plt.plot(time_values, prediction_closing_prices, color='green', label='Predicted Prices')
   
    # Set labels and title
    plt.title("{} Price Prediction".format("BTC"))
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc='upper left')
    
    # Configure the x-axis to display dates appropriately
    # plt.gca().xaxis.set_major_formatter(plt.DateFormatter('%Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Adjust number of ticks as needed
   
    # Show the plot
    plt.show()
    
    # variation 1-2: difference graph
    diff = actual_closing_prices - prediction_closing_prices
    plt.figure(figsize=(16,8))
    plt.plot(time_values, diff)
    plt.title('Difference between Actual and Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price Difference')
    
    # Show the plot
    plt.show()
    
    # variation 1-3: difference median, avarage, etc
    average_diff = np.mean(diff)
    median_diff = np.median(diff)
    print(f"Average difference: {average_diff}")
    print(f"Median difference: {median_diff}")
    
    ####### variation 2: test with given regression_step ########
    
    ## variation 2-1: comparing graphs itself
    ## variation 2-2: difference graph
    ## variation 2-3: difference median, avarage, etc
    
    
    raise NotImplementedError()






#################################################

start_date = dt.datetime(2021, 1, 1)
end_date = dt.datetime(2023, 12, 24)
#end_date = dt.datetime.now()

get_data(1,start_date,  end_date)


