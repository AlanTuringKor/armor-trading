import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import yfinance as yf
import tensorflow as tf
import requests

from model import MyModel
from data import MyData
from my_enum import Feature, ModelType
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.metrics import r2_score





def _visualize_training_history(history):
    historyForPlot = pd.DataFrame(history.history)
    historyForPlot.index += 1 # we plus 1 to the number of indexing so our epochs Plot picture will be counting from 1 not 0.
    historyForPlot.plot()
    plt.ylabel("loss")
    plt.xlabel("epochs")


def test_and_visualize_model(model:MyModel, data:MyData, regression_steps):
    # Create time values for the x-axis
    
    num_data_points = len(data.get_x_data())  # Assuming actual_prices and prediction_prices have the same length
    time_values = [data.get_start_date() + timedelta(days=i) for i in range(num_data_points)]
    
    ####### varition 1 :test with regression_steps of 1 #######
    prediction_values= model.get_model().predict(data.get_x_data())
    prediction_values = data.get_x_scaler().inverse_transform(prediction_values)
    actual_values = data.get_y_data() 
    
    for i in range(0, len(data.get_y_features())):
        if data.get_y_features()[i] == Feature.CLOSING:
            plt.figure(figsize=(16,8))
            plt.plot(actual_values[:,i], color='black', label='Actual Closing Prices')
            plt.plot( prediction_values[:,i], color='green', label='Predicted Closing Prices')
            plt.title("{} Price Prediction".format("BTC"))
            plt.xlabel("Date")
            plt.ylabel("Closing Price")
            plt.legend(loc='upper left')
            plt.show()
            
            # variation 1-2: difference graph
            diff = actual_values[:,i] - prediction_values[:,i]
            plt.figure(figsize=(16,8))
            plt.plot(time_values, diff)
            plt.title('Difference between Actual and Predicted Closing Prices')
            plt.xlabel('Time')
            plt.ylabel('Closing Price Difference')
            
            # Show the plot
            plt.show()
            
            # variation 1-3: difference median, avarage, etc
            average_diff = np.mean(diff)
            median_diff = np.median(diff)
            print(f"Closing Price Average difference: {average_diff}")
            print(f"Closing Price Median difference: {median_diff}")
            
        if data.get_y_features()[i] == Feature.VOLUME:
            plt.figure(figsize=(16,8))
            plt.plot(time_values, actual_values[:,i], color='black', label='Actual Volume')
            plt.plot(time_values, prediction_values[:,i], color='green', label='Predicted Volume')
            plt.title("{} Volume Prediction".format("BTC"))
            plt.xlabel("Date")
            plt.ylabel("Volume")
            plt.legend(loc='upper left')
            plt.show()
            
            # variation 1-2: difference graph
            diff = actual_values[:,i] - prediction_values[:,i]
            plt.figure(figsize=(16,8))
            plt.plot(time_values, diff)
            plt.title('Difference between Actual and Predicted Volume')
            plt.xlabel('Time')
            plt.ylabel('Volume Difference')
            
            # Show the plot
            plt.show()
            
            # variation 1-3: difference median, avarage, etc
            average_diff = np.mean(diff)
            median_diff = np.median(diff)
            print(f"Volume Average difference: {average_diff}")
            print(f"Volume Median difference: {median_diff}")
            
        if data.get_y_features()[i] == Feature.HASHRATE:
            plt.figure(figsize=(16,8))
            plt.plot(time_values, actual_values[:,i], color='black', label='Actual Hashrate')
            plt.plot(time_values, prediction_values[:,i], color='green', label='Predicted Hashrate')
            plt.title("{} Hashrate Prediction".format("BTC"))
            plt.xlabel("Date")
            plt.ylabel("Hashrate")
            plt.legend(loc='upper left')
            plt.show()
            
            # variation 1-2: difference graph
            diff = actual_values[:,i] - prediction_values[:,i]
            plt.figure(figsize=(16,8))
            plt.plot(time_values, diff)
            plt.title('Difference between Actual and Predicted Hashrate')
            plt.xlabel('Time')
            plt.ylabel('Hashrate Difference')
            
            # Show the plot
            plt.show()
            
            # variation 1-3: difference median, avarage, etc
            average_diff = np.mean(diff)
            median_diff = np.median(diff)
            print(f"Closing Price Average difference: {average_diff}")
            print(f"Closing Price Median difference: {median_diff}")
        if data.get_y_features()[i] == Feature.TRANSACTIONCOUNT:
            plt.figure(figsize=(16,8))
            plt.plot(time_values, actual_values[:,i], color='black', label='Actual Transaction Count')
            plt.plot(time_values, prediction_values[:,i], color='green', label='Predicted Transaction Count')
            plt.title("{} Transaction Count Prediction".format("BTC"))
            plt.xlabel("Date")
            plt.ylabel("Transaction Count")
            plt.legend(loc='upper left')
            plt.show()
            
            # variation 1-2: difference graph
            diff = actual_values[:,i] - prediction_values[:,i]
            plt.figure(figsize=(16,8))
            plt.plot(time_values, diff)
            plt.title('Difference between Actual and Predicted Transaction Count')
            plt.xlabel('Time')
            plt.ylabel('Transaction Count Difference')
            
            # Show the plot
            plt.show()
            
            # variation 1-3: difference median, avarage, etc
            average_diff = np.mean(diff)
            median_diff = np.median(diff)
            print(f"Transaction Count Average difference: {average_diff}")
            print(f"Transaction Count Median difference: {median_diff}")
        
    
    # variation 1-2: difference graph
    
    
    ####### variation 2: test with given regression_step ########
    if len(data.get_x_features()) == len(data.get_y_features()):
        ## variation 2-1: comparing graphs itself
        ## variation 2-2: difference graph
        ## variation 2-3: difference median, avarage, etc
        pass

    


#################################################

start_date = dt.datetime(2021, 1, 1)
end_date = dt.datetime(2023, 12, 24)
#end_date = dt.datetime.now()

test_data = MyData(start_date, end_date, 
                   [Feature.CLOSING, Feature.VOLUME, Feature.HASHRATE, Feature.TRANSACTIONCOUNT],
                   [Feature.CLOSING, Feature.VOLUME, Feature.HASHRATE, Feature.TRANSACTIONCOUNT])


test_model = MyModel(ModelType.LSTM,[64,64,64,64,64],test_data)

test_and_visualize_model(test_model, test_data, None)
print('its over')