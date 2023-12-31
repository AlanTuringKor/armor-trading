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





def _visualize_training_history(history):
    historyForPlot = pd.DataFrame(history.history)
    historyForPlot.index += 1 # we plus 1 to the number of indexing so our epochs Plot picture will be counting from 1 not 0.
    historyForPlot.plot()
    plt.ylabel("loss")
    plt.xlabel("epochs")


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
    
    
    pass




#################################################

start_date = dt.datetime(2021, 1, 1)
end_date = dt.datetime(2023, 12, 24)
#end_date = dt.datetime.now()

get_data(1,start_date,  end_date)


