import pandas as pd
import numpy as np
import quandl 
import quadprog
import matplotlib.pyplot as plt


def get_tickers(data):
    return data.columns
    
def get_insample_mean_est(data):
    return data.mean()

def get_insample_cov_est(data):
    return data.cov()

def get_equal_weight(tickers):
    ncol = len(tickers)
    weight = [1/float(ncol) for i in range(ncol)]
    return dict(zip(tickers, weight))

def get_markowitz_weight(tickers, est_mean, est_cov, risk_level=1, short=False):    
    # Sum of weights is 1
    A = np.ones((len(tickers),1))
    B = np.ones(1)
    
    # All weights should be positive
    if ~short:
        A2 = np.identity(len(tickers))
        B2 = 1e-15*np.ones(len(tickers))
        
        A = np.hstack([A, A2])
        B = np.hstack([B, B2])
    
    if risk_level == 0:
        weight = [1 if ticker == est_mean.argmax() else 0 for ticker in tickers]
    elif risk_level == 'inf':
        weight = quadprog.solve_qp( est_cov.values, 0*est_mean.values, A, B, 1 )[0]
    else:
        weight = quadprog.solve_qp( risk_level*est_cov.values, est_mean.values, A, B, 1 )[0]
    return dict(zip(tickers, weight))


def backtest(data, weight, legend=None):
    newdata = data.copy()
    newdata['Portfolio'] = np.sum([weight[col] * newdata[col].values for col in newdata.columns ], axis=0)
    newdata['Return'] = newdata['Portfolio'].pct_change()
    newdata.iat[0, -1] = 0
    newdata['CompoundReturn'] = np.cumprod(1 + newdata['Return'])
    return newdata
   
if __name__ == "__main__":
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
    data4 = data3.pct_change()
    data4 = data4.drop(data4.index[0], axis=0)

    insample_price = data3.iloc[:1500]
    outsample_price = data3.iloc[1500:]

    insample_ret = data4.iloc[:1500]
    outsample_ret = data4.iloc[1500:]

    tickers = get_tickers(data4)
    est_mean = get_insample_mean_est(insample_ret)
    est_cov = get_insample_cov_est(insample_ret)

    eqw = get_equal_weight(tickers)
    p_eqw = backtest(insample_price, eqw)

    w1 = get_markowitz_weight(tickers, est_mean, est_cov, risk_level=1)
    p1 = backtest(insample_price, w1)

    w2 = get_markowitz_weight(tickers, est_mean, est_cov, risk_level=0.01)
    p2 = backtest(insample_price, w2)

    w3 = get_markowitz_weight(tickers, est_mean, est_cov, risk_level=10000)
    p3 = backtest(insample_price, w3)

    w4 = get_markowitz_weight(tickers, est_mean, est_cov, risk_level=0)
    p4 = backtest(insample_price, w4)

    w5 = get_markowitz_weight(tickers, est_mean, est_cov, risk_level='inf')
    p5 = backtest(insample_price, w5)


    fig, ax = plt.subplots(6,1)
    ax[0].bar(eqw.keys(), eqw.values(), label='EQW')
    ax[1].bar(w1.keys(), w1.values(), label='L=1')
    ax[2].bar(w4.keys(), w4.values(), label='L=0')
    ax[3].bar(w2.keys(), w2.values(), label='L=0.01')
    ax[4].bar(w5.keys(), w5.values(), label='L=inf')
    ax[5].bar(w3.keys(), w3.values(), label='L=10000')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    ax[4].legend()
    ax[5].legend()
    plt.show()

    plt.figure()
    plt.plot(p_eqw.index, p_eqw['CompoundReturn'], label='EQW')
    plt.plot(p1.index, p1['CompoundReturn'], label='L=1')
    plt.plot(p4.index, p4['CompoundReturn'], label='L=0')
    plt.plot(p5.index, p5['CompoundReturn'], label='L=inf')
    plt.legend(loc=0)
    plt.grid()
    plt.show()
