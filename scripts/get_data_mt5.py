""" Extracting data directly from MetaTrader5"""

# Import all the necessary libraries!

import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5

pd.set_option('display.max_columns', 500) # number of columns to be displayed
pd.set_option('display.width', 1500)      # max table width to display


def get_mt5_data(currency_symbol = "XAUUSD", timeframe_val= mt5.TIMEFRAME_D1):

    """ This function extracts stock or currency data from mt5 terminal:
    the function needs 2 inputs:
    1. currency_symbol: eg: "XAUUSD" "USDEUR"
    2. timeframe_val: resolution of the data, that could be daily price, 4H(4 hour) price, 1H etc 
                        eg:'mt5.TIMEFRAME_D1' for daily price
                            mt5.TIMEFRAME_H4 for hour 4 price """

    # mt5 initialization
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    
    # getting currency/stock values from mt5 terminal
    rates = mt5.copy_rates_from_pos(currency_symbol, timeframe_val, 0, 10000)
    
    # once extracted, shutdown mt5 session
    mt5.shutdown()

    # dumping data from mt5 to pyndas dataframe
    rates_frame = pd.DataFrame(rates)
    # convert time in seconds into the datetime format
    rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
    
    rates_frame.to_csv('../data/gold_mt5.csv')
    # display data
    print("\nDisplay dataframe with data")
    print(rates_frame)    
    print(filename_)

if __name__=='__main__':

    currency_symbol = "XAUUSD"
    timeframe_val= mt5.TIMEFRAME_D1


    get_mt5_data(currency_symbol,timeframe_val)