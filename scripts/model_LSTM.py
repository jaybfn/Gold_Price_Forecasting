# importing all the necessary libraries!

import os
import warnings
from datetime import datetime 
import pandas as pd
import numpy as np
from numpy.random import seed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = 'white'
warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# function to create all the necessary directory!

def create_dir(path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path, exist_ok = False)
            print("New directory is created")

# load and format data

class DataFormatting():
      
    def __init__(self):
       
        self.df_data = None
        self.df_datetime = None

    def dataset(df):

        # converting time colum from object type to datetime format
        df['time'] = pd.to_datetime(df['time'])
        # splitting the dataframe in to X and y 
        df_data = df[['open','high','low','close','tick_volume']]
        df_datetime =df[['time']]

        return df_data, df_datetime


# split the dataset in tain and test!

def train_test_split(data, train_split=0.9):
    
    """ This function will split the dataframe into training and testing set.
    Inputs: data: Pandas DatFrame
            train_split: default is set to 0.9. Its a ratio to split the trining and testing datset.
    """
    split = int(train_split*len(data)) # for training
    split_test = int(0.90*len(data))
    X_train = data.iloc[:split,:]
    X_val = data.iloc[split:split_test,:]
    X_test = data.iloc[split_test:,:]

    return X_train, X_val, X_test

# Normalize the dataset using standard scaler

class Normalize():
    
    """ class Normalize uses standard scaler method to normalize the dataset"""
    def __init__(self):

        self.data_fit_transformed = None
        self.data_inverse_transformed = None

    def fit_transform(self, data_train, data_val, data_test):

        # initialize StandartScaler()
        scaler = StandardScaler()
        # define transformer
        transformer = [('standard_scaler', StandardScaler(),['open','high','low','close','tick_volume'])]
        # define column transformer
        column_transformer = ColumnTransformer(transformers = transformer)
        # fit and transform training data
        data_fit_transformed = column_transformer.fit_transform(data_train)
        # transform val and test data
        val_transformed = column_transformer.transform(data_val)
        test_transformed = column_transformer.transform(data_test)
        return data_fit_transformed, val_transformed, test_transformed

    def inverse_transform(self, data):

        # initialize StandartScaler()
        scaler = StandardScaler()
        # inverse transform the dataset
        data_inverse_transformed = scaler.inverse_transform(data)
        
        return data_inverse_transformed


# Data transformation (changing data shape to model requirement)

def data_transformation(data, lags = 5):
    
    """ this function transforms dataframe to required input shape for the model.
    It required 2 input arguments:
    1. data: this will be the pandas dataframe
    2. lags: how many previous price points to be used to predict the next future value, in
    this case the default is set to 5 for 'XAUUSD' commodity"""

    # initialize lists to store the dataset
    X_data = []
    y_data = []
    
    for i in range(lags, len(data)):
        X_data.append(data[i-lags: i, 0: data.shape[1]])
        y_data.append(data[i,3:4]) # extracts close price with specific lag as price to be predicted.

    # convert the list to numpy array

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data

# model building

class LSTM_model():
    

    def __init__(self,n_hidden_layers, units, dropout, train_data_X, train_data_y, epochs):

        self.n_hidden_layers = n_hidden_layers
        self.units = units
        self.dropout = dropout
        self.train_data_X = train_data_X
        self.train_data_y = train_data_y
        self.epochs = epochs

    def build_model(self):
        
        model = Sequential()
        # first lstm layer
        model.add(LSTM(self.units, activation='relu', input_shape=(self.train_data_X.shape[1], self.train_data_X.shape[2]), return_sequences=True))
        # building hidden layers
        for i in range(1, self.n_hidden_layers):
            # for the last layer as the return sequence is False
            if i == self.n_hidden_layers -1:
                model.add(LSTM(int(self.units/(2**i)),  activation='relu', return_sequences=False))
            else:
                model.add(LSTM(int(self.units/(2**i)),  activation='relu', return_sequences=True))
        # adding droupout layer
        model.add(Dropout(self.dropout))
        # final layer
        model.add(Dense(self.train_data_y.shape[1]))
        return model


if __name__ == '__main__':
    
    seed(42)
    tf.random.set_seed(42) 
    keras.backend.clear_session()

    # model hyperparameters!
    n_hidden_layers = 3
    units = 128
    dropout = 0.2
    epochs = 1

    # creating main folder
    today = datetime.now()
    today  = today.strftime('%Y_%m_%d')
    path = '../Model_Outputs/'+ today
    create_dir(path)
 
    # creating directory to save model and its output
    folder = 'model_lstm'+ str(units)
    path_main = path + '/'+ folder
    create_dir(path_main)

    # creating directory to save all the metric data
    folder = 'metrics'
    path_metrics = path_main +'/'+ folder
    create_dir(path_metrics)

    # creating folder to save model.h5 file
    folder = 'model'
    path_model = path_main +'/'+ folder
    create_dir(path_model)

    # creating folder to save model.h5 file
    folder = 'model_checkpoint'
    path_checkpoint = path_main +'/'+ folder
    create_dir(path_checkpoint)

    # loading the dataset!
    data = pd.read_csv('../data/gold_mt5.csv',index_col=[0]) 

    # initializing DataFormatting class
    data_init = DataFormatting()
    df_data, _ = DataFormatting.dataset(data)
    print('\n')
    print('Displaying top 5 rows of the dataset:')
    print('\n')
    print(df_data.head())

    # create train test split

    X_train, X_val , X_test = train_test_split(df_data, train_split=0.9)

    # normalize train, val and test dataset
    scaler_init = Normalize()
    data_fit_transformed, val_transformed, test_transformed = scaler_init.fit_transform(X_train, X_val, X_test)
    print('\n')
    print('Displaying top 5 rows of all the scaled dataset:')
    print('\n')
    print(data_fit_transformed[0:5], val_transformed[0:5], test_transformed[0:5])

    
    # changing shape of the data to match the model requirement!

    X_data, y_data = data_transformation(data_fit_transformed, lags = 5)
    print('\n')
    print('Displaying the shape of the dataset required by the model:')
    print('\n')
    print(f' Input shape X:',X_data.shape, f'Input shape y:',y_data.shape)
    print('\n')

    X_val_data, y_val_data = data_transformation(val_transformed, lags = 5)
    X_test_data, y_test_data = data_transformation(test_transformed, lags= 5)

    # input data
    train_data_X = X_data 
    train_data_y = y_data

    # initializing model
    model_init = LSTM_model(n_hidden_layers, units, dropout, train_data_X, train_data_y, epochs)

    # calling the model
    model = model_init.build_model()

    # metrics for evaluating the model
    metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()]

    # model compiler
    model.compile(optimizer=Adam(learning_rate = 0.0001), loss='mse', metrics = metrics)

    # setting the model file name
    model_name = 'lstm_'+ str(units)+'.h5'
    
    # setting the callback function
    cb = [
        tf.keras.callbacks.ModelCheckpoint(path_checkpoint),
        tf.keras.callbacks.CSVLogger(path_metrics+'/'+'data.csv'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)]

    # model fitting protocol
    history = model.fit(train_data_X,train_data_y, 
                        epochs = epochs, 
                        batch_size = 8, 
                        validation_data=(X_val_data, y_val_data), 
                        verbose = 1,
                        callbacks=[cb],
                        shuffle= False)

    # Model evaluation

    # training dataset
    train_loss, RMSE, MAE, MAPE = model.evaluate(train_data_X,train_data_y)
    print('\n','Evaluation of Training dataset:','\n','train_loss:',round(train_loss,3),'\n','RMSE:',round(RMSE,3),'\n', 'MAE:',round(MAE,3),'\n','MAPE:',round(MAPE,3))
    
    # validation dataset
    val_loss, val_RMSE, val_MAE, val_MAPE = model.evaluate(X_val_data, y_val_data)
    print('\n','Evaluation of Validation dataset:','\n','train_loss:',round(val_loss,3),'\n','val_RMSE:',round(val_RMSE,3),'\n', 'val_MAE:',round(val_MAE,3),'\n','MAPE:',round(MAPE,3))
    # path to save model
    
    model.save(path_model+'/'+model_name)   



