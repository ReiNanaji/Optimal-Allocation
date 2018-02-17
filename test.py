
import pandas as pd
import numpy as np
import quandl 


quandl.ApiConfig.api_key = '8h7yyk7CKS3fd6ikTuLB'



data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', header=0)[0]



tickers = data['Ticker symbol'].tolist()


test = quandl.get_table('WIKI/PRICES', 
                        qopts={ 'columns': [ 'ticker', 'date', 'close' ] },
                        date={ 'gte': '2010-01-01', 'lte': '2018-01-01' }, 
                        ticker=tickers[0:20],
                        paginate=True)



df = test.groupby('ticker')['date'].count().reset_index()
tickers_to_keep = df.loc[df['date'] == 2012, 'ticker'].tolist()         

data2 = quandl.get_table('WIKI/PRICES', 
                        qopts={ 'columns': [ 'ticker', 'date', 'close' ] },
                        date={ 'gte': '2010-01-01', 'lte': '2018-01-01' }, 
                        ticker=tickers_to_keep,
                        paginate=True)


data3 = pd.pivot_table(data=data2, index='date', columns='ticker', values='close')

insample = data3.iloc[:1500]
outsample = data3.iloc[1500:]


def get_equal_weight(data):
    ncol = len(data.columns)
    tickers = data.columns
    weight = [1/float(ncol) for i in range(ncol)]
    return dict(zip(tickers, weight))


def get_markowitz_weight(data):
    est_mean = data.mean()
    est_cov = data.cov()
    return est_mean, est_cov


def backtest(data, weight):
    newdata = data.copy()
    newdata['Portfolio'] = np.sum([weight[col] * newdata[col].values for col in newdata.columns ], axis=0)
    newdata['Return'] = newdata['Portfolio'].pct_change()
    newdata.iat[0, -1] = 0
    newdata['CompoundReturn'] = np.cumprod(1 + newdata['Return'])
    newdata['CompoundReturn'].plot()
    return newdata
   
    
weight = get_equal_weight(insample) 
portfolio = backtest(outsample, weight)

mean, cov = get_markowitz_weight(insample)
