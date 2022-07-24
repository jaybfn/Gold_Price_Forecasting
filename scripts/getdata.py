# importing necessary library
import pandas_datareader as web

# defining a preprocessing function for the yahoofinance dataet:

def preprocessing(symbol):
    
    # Importing the dataset:
    df = web.DataReader(symbol, data_source='yahoo',start='2000-08-30', end='2022-07-22').dropna()
    
    # removing the adjclose columns:
    del df['Adj Close']

    return df

if __name__=='__main__':

    # Download the dataset from yfinance
    XAUUSD = 'GC=F'
    df = preprocessing(XAUUSD)
    print(df)