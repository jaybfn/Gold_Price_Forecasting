# Exp1 : mlflow ui --backend-store-uri sqlite:///lstm_EU.db

# importing all the necessary libraries!

import os
import warnings
import json
from datetime import datetime 
import pandas as pd
import numpy as np
from math import sqrt
from numpy.random import seed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from keras.regularizers import L1L2
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor

import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
plt.rcParams['figure.facecolor'] = 'white'
warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# function to create all the necessary directory!

def create_dir(path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path, exist_ok = False)
            print("New directory is created")


# dumping all the hyperparameters to json file!
def hyperparms(dictionary):
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)
    
    # Writing to sample.json
    with open(path_metrics +'/'+ 'hyperparm.json', "w") as outfile:
        outfile.write(json_object)

#load and format data

class DataFormatting():
      
    def __init__(self):
       
        self.df_data = None
        self.df_datetime = None

    def dataset(df):

        # converting time colum from object type to datetime format
        df['date'] = pd.to_datetime(df['date'],dayfirst = True, format = '%Y-%m-%d')
        df = df.dropna()
        # splitting the dataframe in to X and y 
        df_data = df[['open','high','low','close']] #'high','low',,'CRUDE_OIL_CLOSE','US500_CLOSE','open','EXCHANGE_RATE',
        df_datetime =df[['date']]

        return df_data, df_datetime


# Data transformation (changing data shape to model requirement)

def data_transformation(data, lags = 5, n_fut = 1):
    
    """ this function transforms dataframe to required input shape for the model.
    It required 2 input arguments:
    1. data: this will be the pandas dataframe
    2. lags: how many previous price points to be used to predict the next future value, in
    this case the default is set to 5 for 'XAUUSD' commodity"""

    # initialize lists to store the dataset
    X_data = []
    y_data = []
    
    for i in range(lags, len(data)- n_fut +1):
        X_data.append(data[i-lags: i, 0: data.shape[1]])
        y_data.append(data[i+ n_fut-1:i+n_fut,3]) # extracts close price with specific lag as price to be predicted.

    # convert the list to numpy array

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data


def build_model(space): #train_data_X.shape[1], train_data_X.shape[2]

    with mlflow.start_run():
        mlflow.set_tag('model','lstm')
        mlflow.log_params(space)
        
        model = Sequential()
            # first lstm layer
        model.add(LSTM(units = space['units'], activation='tanh', input_shape=(train_data_X.shape[1],train_data_X.shape[2]), kernel_regularizer=L1L2(l1=space['l1'], l2=space['l2']), return_sequences=True))

        if space['layers'] !=1:

            # building hidden layers
            for i in range(1, space['layers']):
                # for the last layer as the return sequence is False
                if i == space['layers'] -1:
                    model.add(LSTM(int(space['units']/(2**i)),  activation='tanh', return_sequences=False))
                else:
                    model.add(LSTM(int(space['units']/(2**i)),  activation='tanh', return_sequences=True))

        else:
            model.add(LSTM(int(space['units']/2),  activation='tanh', return_sequences=False))

        # adding dropout layer
        model.add(Dropout(space['dropout']))
        # final layer
        model.add(Dense(train_data_y.shape[1])) #train_data_y.shape[1]
        # metrics for evaluating the model
        metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()]

        # model compiler
        model.compile(optimizer=Adam(learning_rate = space['rate']), loss='mse', metrics = metrics)

        # setting the callback function
        cb = [
            tf.keras.callbacks.ModelCheckpoint(path_checkpoint),
            tf.keras.callbacks.CSVLogger(path_metrics+'/'+'data.csv')]

        # model fitting protocol
        history = model.fit(x_train, y_train, 
                            epochs = 700, 
                            batch_size = space['batch_size'],  
                            validation_data=(x_test, y_test),
                            verbose = 1,
                            callbacks=[cb],
                            shuffle= False,
                            workers = 16,
                            use_multiprocessing = True)
        
        # training dataset
        train_loss, RMSE, MAE, MAPE = model.evaluate(x_train, y_train)
        print('\n','Evaluation of Training dataset:','\n''\n','train_loss:',round(train_loss,3),'\n','RMSE:',round(RMSE,3),'\n', 'MAE:',round(MAE,3),'\n','MAPE:',round(MAPE,3))
        
        #val dataset
        val_loss, val_RMSE, val_MAE, val_MAPE = model.evaluate(x_test, y_test)
        print('\n','Evaluation of Testing dataset:','\n''\n','val_loss:',round(val_loss,3),'\n','val_RMSE:',round(val_RMSE,3),'\n', 'val_MAE:',round(val_MAE,3),'\n','val_MAPE:',round(val_MAPE,3))
                
        
        mlflow.log_metric('train_loss',train_loss)
        mlflow.log_metric('RMSE', RMSE)
        mlflow.log_metric('MAE', MAE)
        mlflow.log_metric('MAPE',MAPE)

        mlflow.log_metric('val_loss',val_loss)
        mlflow.log_metric('val_RMSE', val_RMSE)
        mlflow.log_metric('val_MAE', val_MAE)
        mlflow.log_metric('val_MAPE',val_MAPE)


    
        return {'loss': MAPE, 'val_loss': val_MAPE, 'status': STATUS_OK, 'model': model, 'space':space}


class Bi_LSTM_model():
    

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
        model.add(Bidirectional(LSTM(self.units, activation='tanh', input_shape=(self.train_data_X.shape[1], self.train_data_X.shape[2]), kernel_regularizer=self.reg, return_sequences=True)))
        # building hidden layers
        
        if self.n_hidden_layers !=1:

            for i in range(1, self.n_hidden_layers):
                # for the last layer as the return sequence is False
                if i == self.n_hidden_layers -1:
                    model.add(Bidirectional(LSTM(int(self.units/(2**i)),  activation='tanh', return_sequences=False)))
                else:
                    model.add(Bidirectional(LSTM(int(self.units/(2**i)),  activation='tanh', return_sequences=True)))

        else:
            model.add(Bidirectional(LSTM(int(self.units/2),  activation='tanh', return_sequences=False)))
        
        # adding dropout layer
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
    #DATEBASE_NAME = input('Enter new database name:')
    mlflow.set_tracking_uri("sqlite:///lstm_EU.db")
    mlflow.set_experiment("Currency_Price_Forecasting")

    data = pd.read_csv('../data/macro_data/EURUSD.csv',index_col=[0]) 

    seed(42)
    tf.random.set_seed(42) 
    keras.backend.clear_session()

    # hyperparameters
    lag = 1
    n_fut = 1
    train_size_percent = 90

    # creating main folder
    today = datetime.now()
    today  = today.strftime('%Y_%m_%d')
    path = '../Model_Outputs/'+ today
    create_dir(path)

    
    # creating directory to save model and its output
    EXPERIMENT_NAME = input('Enter new Experiment name:')
    print('\n')
    print('A folder with',EXPERIMENT_NAME,'name has be created to store all the model details!')
    print('\n')
    folder = EXPERIMENT_NAME
    path_main = path + '/'+ folder
    create_dir(path_main)

    # creating directory to save model and its output
    folder = 'model_Bilstm'#+ str(units) + '_' + str(n_hidden_layers)
    path_dir = path_main + '/'+ folder
    create_dir(path_dir)

    # creating directory to save all the metric data
    folder = 'metrics'
    path_metrics = path_dir +'/'+ folder
    create_dir(path_metrics)

    # creating folder to save model.h5 file
    folder = 'model'
    path_model = path_dir +'/'+ folder
    create_dir(path_model)

    # creating folder to save model.h5 file
    folder = 'model_checkpoint'
    path_checkpoint = path_dir +'/'+ folder
    create_dir(path_checkpoint)

    # creating folder to save model.h5 file
    folder = 'forecasting_resutls'
    path_forecast = path_dir +'/'+ folder
    create_dir(path_forecast)


    # initializing DataFormatting class
    data_init = DataFormatting()
    df_data, df_datetime = DataFormatting.dataset(data)
    print('\n')
    print('Displaying top 5 rows of the dataset:')
    print('\n')
    print(df_data.head())

    # normalize train, val and test dataset

    # initialize StandartScaler()
    scaler = StandardScaler()
    scaler = scaler.fit(df_data)
    data_fit_transformed = scaler.transform(df_data)


    print('\n')
    print('Displaying top 5 rows of all the scaled dataset:')
    print('\n')
    #print('The train dateset:','\n''\n',data_fit_transformed[0:5],'\n''\n', 'The validation dataset:','\n''\n',val_transformed[0:5],'\n''\n','The test dataset:','\n''\n',test_transformed[0:5])
    print('The train dateset:','\n''\n',data_fit_transformed[0:10])

    # changing shape of the data to match the model requirement!

    X_data, y_data = data_transformation(data_fit_transformed, lags = lag, n_fut = n_fut)
    print('\n')
    print('Displaying the shape of the dataset required by the model:')
    print('\n')
    print(f' Input shape X:',X_data.shape, f'Input shape y:',y_data.shape)
    print('\n')
    #print(X_data)
    print(y_data[0:10])
    # # # setting the model file name
    #model_name = 'lstm_'+ str(units)+'.h5'

    # input data
    train_data_X = X_data
    train_data_y = y_data

   # traininig 
    size_df = X_data.shape[0]
    train_size_percent = train_size_percent/100
    train_size = int(round(size_df*train_size_percent,0))
    x_train = X_data[:train_size,:]
    y_train = y_data[:train_size,:]

    # testing

    x_test = X_data[train_size:,:]
    y_test = y_data[train_size:,:]

    print(x_test.shape)
    print(y_test.shape)

    space = {'rate'       : hp.uniform('rate',0.00001,0.001),
            'units'      : scope.int(hp.quniform('units',32,256,32)),
            'batch_size' : scope.int(hp.quniform('batch_size',8,128,16)),
            'layers'     : scope.int(hp.quniform('layers',1,6,1)),
            'dropout'     : hp.quniform('dropout',0,0.75,0.25),
            'l1': hp.uniform('l1', 0.001, 0.1),
            'l2': hp.uniform('l2', 0.01,0.1)}

    mlflow.keras.autolog()

    best_result = fmin(
        fn = build_model,
        space = space,
        algo = tpe.suggest,
        max_evals = 50,
        trials = Trials()
        )
