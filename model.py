import tensorflow as tf

from data import MyData
from my_enum import Feature, ModelType
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.models import Sequential


# TODO: Does callback function work?


class MyModel:
    def __init__(self, model_type:ModelType, layers, data:MyData):
        self._create_and_fit_model(model_type,layers,data.get_x_data(), data.get_y_data())
    
    def get_model(self):
        return self.model
    
    def get_history(self):
        return self.history
    
    # this function also should be transfers to model class. only prediction and visualization are suiting for this script
    def _create_and_fit_model(self, model_type, layers, x_data, y_data):
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
        if model_type == ModelType.LSTM:
            return self._create_and_fit_model_LSTM(layers, x_data, y_data)
        elif model_type == ModelType.GRU:
            return self._create_and_fit_model_GRU(layers, x_data, y_data)
        elif model_type == ModelType.BILSTM:
            return self._create_and_fit_model_BILSTM(layers, x_data, y_data)
        else:
            print("wrong input bro! only one of these (LSTM, GRU, BILSTM)")

    

    def _create_and_fit_model_LSTM(self, layers, x_data, y_data):
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

        self.model = model
        self.history = history



    def _create_and_fit_model_BILSTM(self, layers, x_data, y_data):
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

        self.model = model
        self.history = history

    def _create_and_fit_model_GRU(self, layers, x_data, y_data):
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
        
        self.model = model
        self.history = history