#Tutorial used from https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


from collections import deque
import numpy as np
import pandas as pd
import random
from tensorflow.keras.layers import LSTM

np.random.seed(3)
tf.random.set_seed(314)
random.seed(314)

coin_id=['BTC','ETH','ADA','XRP', 'USDT','BCH','LINK','LTC','BNB','EOS','XLM','TRX' ]
#coin_id=['TRX','XEM','XMR','MIOTA', 'VET','DASH','ETC','ZEC','OMG','BAT','DOGE','ZRX','WAVES','DGB','KNC','ICX','LRC','QTUM','REP','LSK','ANT','DCR','BTG','SC','NANO','BNT','BCS','SNT' ]
# parameters
N_STEPS = 80
# Lookup step, 1 is the next day
LOOKUP_STEP = 60
# test ratio size
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")

### model parameters

N_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 500

# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")

if not os.path.isdir("logs"):
    os.mkdir("logs")

if not os.path.isdir("data"):
    os.mkdir("data")
if not os.path.isdir("model"):
    os.mkdir("model")




for i in coin_id:
    ticker = i+"-USD"
    ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
    # model name to save, making it as unique as possible based on parameters
    model_name = f"{date_now}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
    if BIDIRECTIONAL:
        model_name += "-b"

    def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1,
                  test_size=0.2, feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):

        #Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.

        # see if ticker is already a loaded stock from yahoo finance, if it alreay loaded use it directly, else load it from the yahoo_fin library
        if isinstance(ticker, str):
            df = si.get_data(ticker)
        elif isinstance(ticker, pd.DataFrame):
            df = ticker
        else:
            raise TypeError("error, cannot load ticker")

        result = {}
        result['df'] = df.copy()
        for i in feature_columns:
            assert i in df.columns, f"'{i}' does not exist in the dataframe."

        if scale:
            column_scaler = {}
            # scale the data (prices) from 0 to 1
            for column in feature_columns:
                data_scaler = preprocessing.MinMaxScaler()
                df[column] = data_scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
                column_scaler[column] = data_scaler

            # add the MinMaxScaler instances to the result returned
            result["column_scaler"] = column_scaler

        df['future'] = df['adjclose'].shift(-lookup_step)

        # last `lookup_step` columns contains NaN in future column
        # get them before droping NaNs
        last_sequence = np.array(df[feature_columns].tail(lookup_step))

        # drop NaNs
        df.dropna(inplace=True)

        sequence_data = []
        sequences = deque(maxlen=n_steps)

        for entry, target in zip(df[feature_columns].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == n_steps:
                sequence_data.append([np.array(sequences), target])

        # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
        # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
        # this last_sequence will be used to predict in future dates that are not available in the dataset
        last_sequence = list(sequences) + list(last_sequence)
        # shift the last sequence by -1
        last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
        # add to result
        result['last_sequence'] = last_sequence

        # construct the X's and y's
        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)

        # convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # reshape X to fit the neural network
        X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))

        # split the dataset
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                    test_size=test_size,
                                                                                           shuffle=shuffle)
        # return the result
        return result
    # load the data


    def create_model(sequence_length, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                     loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
        model = Sequential()
        for i in range(n_layers):
            if i == 0:
                # first layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True), input_shape=(None, sequence_length)))
                else:
                    model.add(cell(units, return_sequences=True, input_shape=(None, sequence_length)))
            elif i == n_layers - 1:
                # last layer
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=False)))
                else:
                    model.add(cell(units, return_sequences=False))
            else:
                # hidden layers
                if bidirectional:
                    model.add(Bidirectional(cell(units, return_sequences=True)))
                else:
                    model.add(cell(units, return_sequences=True))
            # add dropout after each layer
            model.add(Dropout(dropout))
        model.add(Dense(1, activation="linear"))
        model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
        return model
    data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)
    data["df"].to_csv(ticker_data_filename)

    model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                         dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

    # some tensorflow callbacks
    checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True,
                                   save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        callbacks=[checkpointer, tensorboard],
                        verbose=1)

    model.save(os.path.join("results", model_name) + ".h5")



