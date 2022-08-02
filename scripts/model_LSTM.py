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
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import L1L2
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
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
        df['date'] = pd.to_datetime(df['date'])
        # splitting the dataframe in to X and y 
        df_data = df[['open','high','low','close']]
        df_datetime =df[['date']]

        return df_data, df_datetime


# split the dataset in tain and test!

def train_test_split(data, train_split=0.7):
    
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
    

    def __init__(self,n_hidden_layers, units, dropout, train_data_X, train_data_y, epochs, reg):

        self.n_hidden_layers = n_hidden_layers
        self.units = units
        self.dropout = dropout
        self.train_data_X = train_data_X
        self.train_data_y = train_data_y
        self.epochs = epochs
        self.reg = reg

    def build_model(self):
        
        model = Sequential()
        # first lstm layer
        model.add(LSTM(self.units, activation='relu', input_shape=(self.train_data_X.shape[1], self.train_data_X.shape[2]), kernel_regularizer=self.reg, return_sequences=True))
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

def metricplot(df, xlab, ylab_1,ylab_2, path):
    
    """
    This function plots metric curves and saves it
    to respective folder
    inputs: df : pandas dataframe 
            xlab: x-axis
            ylab_1 : yaxis_1
            ylab_2 : yaxis_2
            path: full path for saving the plot
            """
    plt.figure()
    sns.set_theme(style="darkgrid")
    sns.lineplot(x = df[xlab], y = df[ylab_1])
    sns.lineplot(x = df[xlab], y = df[ylab_2])
    plt.xlabel('Epochs',fontsize = 12)
    plt.ylabel(ylab_1,fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.legend([ylab_1,ylab_2], prop={"size":12})
    plt.savefig(path+'/'+ ylab_1)
    #plt.show()

if __name__ == '__main__':
    
    seed(42)
    tf.random.set_seed(42) 
    keras.backend.clear_session()

    # model hyperparameters!
    lag = 3
    n_hidden_layers = 2
    batch_size = 128
    units = 128
    dropout = 0.3
    epochs = 300
    reg = L1L2(l1=0.00, l2=0.00)

    # creating main folder
    today = datetime.now()
    today  = today.strftime('%Y_%m_%d')
    path = '../Model_Outputs/'+ today
    create_dir(path)
 
    # creating directory to save model and its output
    folder = 'model_lstm'+ str(units) + '_' + str(n_hidden_layers)
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
    df_data, df_datetime = DataFormatting.dataset(data)
    print('\n')
    print('Displaying top 5 rows of the dataset:')
    print('\n')
    print(df_data.head())

    # create train test split

    X_train, X_val , X_test = train_test_split(df_data, train_split=0.7)

    # normalize train, val and test dataset

    # initialize StandartScaler()
    scaler = MinMaxScaler()
    scaler = scaler.fit(X_train)
    data_fit_transformed = scaler.transform(X_train)
    val_transformed = scaler.transform(X_val)
    test_transformed = scaler.transform(X_test)
    
    print('\n')
    print('Displaying top 5 rows of all the scaled dataset:')
    print('\n')
    print('The train dateset:','\n''\n',data_fit_transformed[0:5],'\n''\n', 'The validation dataset:','\n''\n',val_transformed[0:5],'\n''\n','The test dataset:','\n''\n',test_transformed[0:5])

    
    # changing shape of the data to match the model requirement!

    X_data, y_data = data_transformation(data_fit_transformed, lags = lag)
    print('\n')
    print('Displaying the shape of the dataset required by the model:')
    print('\n')
    print(f' Input shape X:',X_data.shape, f'Input shape y:',y_data.shape)
    print('\n')

    X_val_data, y_val_data = data_transformation(val_transformed, lags = lag)
    X_test_data, y_test_data = data_transformation(test_transformed, lags= lag)

    # input data
    train_data_X = X_data 
    train_data_y = y_data

    # initializing model
    model_init = LSTM_model(n_hidden_layers, units, dropout, train_data_X, train_data_y, epochs, reg)

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
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1001, restore_best_weights=False)]

    # model fitting protocol
    history = model.fit(train_data_X,train_data_y, 
                        epochs = epochs, 
                        batch_size = batch_size, 
                        validation_data=(X_val_data, y_val_data), 
                        verbose = 1,
                        callbacks=[cb],
                        shuffle= False)

    # Model evaluation

    # training dataset
    train_loss, RMSE, MAE, MAPE = model.evaluate(train_data_X,train_data_y)
    print('\n','Evaluation of Training dataset:','\n''\n','train_loss:',round(train_loss,3),'\n','RMSE:',round(RMSE,3),'\n', 'MAE:',round(MAE,3),'\n','MAPE:',round(MAPE,3))
    
    # validation dataset
    val_loss, val_RMSE, val_MAE, val_MAPE = model.evaluate(X_val_data, y_val_data)
    print('\n','Evaluation of Validation dataset:','\n''\n','val_loss:',round(val_loss,3),'\n','val_RMSE:',round(val_RMSE,3),'\n', 'val_MAE:',round(val_MAE,3),'\n','MAPE:',round(MAPE,3))
    # path to save model

    model.save(path_model+'/'+model_name)   

    path_metrics+'/'+'data.csv'
    df = pd.read_csv(path_metrics+'/'+'data.csv')

    metricplot(df, 'epoch', 'loss','val_loss', path_metrics)
    metricplot(df, 'epoch', 'mean_absolute_error','val_mean_absolute_error', path_metrics)
    metricplot(df, 'epoch', 'mean_absolute_percentage_error','val_mean_absolute_percentage_error', path_metrics)
    metricplot(df, 'epoch', 'root_mean_squared_error','val_root_mean_squared_error', path_metrics)



    model_name = 'lstm_'+ str(units)+'.h5'
    model_eval = load_model(path_model+'/'+model_name, compile=False)


    # prediction on the test set
    
    y_pred_close = model_eval.predict(X_test_data)
    y_pred_close_copies = np.repeat(y_pred_close, X_train.shape[1], axis = -1)
    y_pred_scaled = scaler.inverse_transform(y_pred_close_copies)[:,0]

    y_test_true = np.repeat(y_test_data, X_train.shape[1], axis = -1)
    y_test_scaled = scaler.inverse_transform(y_test_true)[:,0]
    
    # lets compare the predicted output and the original values using RMSE as a metric
    
    rmse = tf.keras.metrics.RootMeanSquaredError()
    RMSE_test = rmse(y_test_scaled, y_pred_scaled)
    print('\n')
    print('The RMSE on the test data set is:',RMSE_test)
    print('\n')

    future_days = 10

    forecast_date = pd.date_range(list(df_datetime['date'])[-1], periods = future_days, freq = '1d').tolist()
    forecast = model_eval.predict(X_data[-future_days:])
    #print(forecast)
    forecast_copies = np.repeat(forecast, X_data.shape[2], axis = -1 )
    #print(forecast_copies)
    y_pred_fut = scaler.inverse_transform(forecast_copies)[:,0]
    print('The forecast for the future 10 days is:','\n',y_pred_fut)