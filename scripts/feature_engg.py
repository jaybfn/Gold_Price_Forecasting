#importing all the necessary libraries

#pip install ta

from getdata import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ta
plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')

# defining all the required features in to a single function
def feature_engineering(symbol, price_type):
    
    # get_data
    df = preprocessing(symbol)

    # subset the dataframe
    #df_close_data = df[[price_type]] 
    df_close_data = df.copy()

    # create return for the specific col
    df_close_data['returns'] = df_close_data[price_type].pct_change(1)

    # Create Simple Moving Average (SMA)

    # SMA 15 days
    df_close_data['SMA_15'] = df_close_data[[price_type]].rolling(15).mean().shift(1)

    # SMA 60 days
    df_close_data['SMA_60'] = df_close_data[[price_type]].rolling(60).mean().shift(1)

    # Create Moving Standard Deviation aka Volatility in the price.

    # SMA 15 days
    df_close_data['MSD_15'] = df_close_data[['returns']].rolling(15).std().shift(1)

    # SMA 60 days
    df_close_data['MSD_60'] = df_close_data[['returns']].rolling(60).std().shift(1)

    # create RSI indicator

    RSI = ta.momentum.RSIIndicator(df_close_data[price_type], window=14, fillna = False)
    df_close_data['rsi_14'] = RSI.rsi().shift(1)
    
    return df_close_data


if __name__ == '__main__':

    symbol = 'GC=F'
    price_type = 'Close'
    df_close = feature_engineering(symbol, price_type)
    print(df_close)
