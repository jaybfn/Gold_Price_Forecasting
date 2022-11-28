import os
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

plt.rcParams['figure.facecolor'] = 'white'
warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DataFormatting():
      
    def __init__(self):
        self.df_data = None
        self.df_datetime = None
       
    def dataset(df):

        # converting time colum from object type to datetime format
        df['date'] = pd.to_datetime(df['date'],dayfirst = True, format = '%d/%m/%Y')
        df = df.dropna()
        # splitting the dataframe in to X and y 
        df_data = df[['open','high','low','close']] 
        df_datetime =df[['date']]

        return df_data, df_datetime


# Data transformation (changing data shape to model requirement)

def data_transformation(data, lags = 5, n_fut = 1):
    
    """ this function transforms dataframe to required input shape for the model.
    It required 2 input arguments:
    1. data: this will be the pandas dataframe
    2. lags: how many previous price points to be used to predict the next future value, in
    this case the default is set to 5 for 'EURUSD' commodity"""

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


if __name__ == '__main__':

    # parms
    lag = 5 # number of days in past required ot predit the future                                                                      
    n_fut = 1 # to predict one day in future

    # paths
    SCALING_PATH = 'EU_scaler_std.bin'
    CSV_PATH = '../data/forecast_EURUSD.csv'
    MODEL_PATH = "../Model_Outputs/2022_11_12/EURUSD/model_Bilstm/model/lstm_192.h5"

    # loading scaling model
    std_scaler = joblib.load(SCALING_PATH)

    # loading the dataset 
    data = pd.read_csv(CSV_PATH,index_col=[0]) 

    # initializing DataFormatting class
    data_init = DataFormatting()
    df_data, df_datetime = DataFormatting.dataset(data)
    df_colnames = list(df_data.columns)
    # scaling
    data_fit_transformed = std_scaler.transform(df_data)
    # transforming data for the model 
    X_data, y_data = data_transformation(data_fit_transformed, lags = lag, n_fut = n_fut)
    # laoding the model
    model_eval = load_model(MODEL_PATH, compile=False)
    # predit on the data
    forecast = model_eval.predict(X_data)
    forecast_copies = np.repeat(forecast, df_data.shape[1], axis = -1 )
    y_pred_fut = std_scaler.inverse_transform(forecast_copies)[:,0]
    print(y_pred_fut)