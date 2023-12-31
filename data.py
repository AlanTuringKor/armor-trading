import yfinance as yf
import numpy as np
import requests

from my_enum import Feature
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta



class MyData:
    def __init__(self, start_date, end_date, x_features, y_features):
        self.start_date = start_date
        self.end_date = end_date
        self.x_features = x_features
        self.y_features = y_features    
        self._fetch_data() 

    def get_data(self):
        return self.data
    
    def get_x_scaler(self):
        return self.x_scaler
    
    def get_y_scaler(self):
        return self.y_scaler
    
    def get_x_data(self):
        return self.x_data
    
    def get_y_data(self):
        return self.y_data
        
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
    
    
    def _fetch_data(self):
        df = yf.download(f'BTC-USD', start=self.start_date, end=self.end_date)
    
        print("the most recent date of the data is:" + df.tail(1).index[0].strftime('%Y-%m-%d %H:%M:%S'))

        # Printing the first few entries of the dataset
        print("\n\nThe head of the dataframe of {}: \n\n".format("BTC"))
        print(df.head())

        print("\n\n Description of the dataset of {}: \n\n".format("BTC"))
        print(df.describe())

        # Pre-Processing
        close = df[['Close']].values
        volume = df[['Volume']].values
        hashrate = np.array(self._get_hashrate(self.start_date, self.end_date)).reshape(-1, 1)
        txCount = np.array(self._get_transaction_count(self.start_date, self.end_date)).reshape(-1, 1)
        
        # if the length of those 4 numpy array is not same than delete the latest Element of each list so the length of those list corresponds the length of the shortest list 
        # Find the minimum length among all arrays
        min_length = min(len(close), len(volume), len(hashrate), len(txCount))

        # Truncate each array to the minimum length
        close = close[:min_length]
        volume = volume[:min_length]
        hashrate = hashrate[:min_length]
        txCount = txCount[:min_length]
        
        # Looks back on 60 days of data to predict the values of 61st day
        lookback = 60
        
        # for x_data
        
        x_df = np.empty((0, close.shape[1]))
        
        if Feature.CLOSING in self.x_features:
            x_df = np.vstack((x_df, close))
        if Feature.VOLUME in self.x_features:
            x_df = np.vstack((x_df, volume))
        if Feature.HASHRATE in self.x_features:
            x_df = np.vstack((x_df, hashrate))
        if Feature.TRANSACTIONCOUNT in self.x_features:
            x_df = np.vstack((x_df, txCount))
        
        # Scaling the data using the Min Max Scaling between 0 and 1
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))
        x_df_scaled = self.x_scaler.fit_transform(x_df.reshape(-1, len(self.x_features)))
        x_data = []
        
          
        # Filling up the x_data and y_data with the scaled data
        for i in range(lookback, len(x_df_scaled)):

            x_data.append(x_df_scaled[i - lookback: i, 0:len(self.x_features)])      
        
        # for y_data
        y_df = np.empty((0, close.shape[1]))
        
        if Feature.CLOSING in self.y_features:
            y_df = np.vstack((y_df, close))
        if Feature.VOLUME in self.y_features:
            y_df = np.vstack((y_df, volume))
        if Feature.HASHRATE in self.y_features:
            y_df = np.vstack((y_df, hashrate))
        if Feature.TRANSACTIONCOUNT in self.y_features:
            y_df = np.vstack((y_df, txCount))
        
        # Scaling the data using the Min Max Scaling between 0 and 1
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        y_df_scaled = self.y_scaler.fit_transform(y_df.reshape(-1, len(self.y_features)))
                
        y_data = []
               
        for i in range(lookback, len(y_df_scaled)):
            # The value of Closing price at i is the the required output/label
            y_data.append(x_df_scaled[i, 0:len(self.y_features)])


        # Converting the data set we have created into a numpy array
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        
        
        print("\n\n The number of samples in our training data = " + str(len(x_data)) + "\n\n")
        
        self.x_data = x_data
        self.y_data = y_data
        
