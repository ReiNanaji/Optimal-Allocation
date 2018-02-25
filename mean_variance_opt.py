import pandas as pd
import numpy as np
import quandl 
import quadprog
import matplotlib.pyplot as plt
from fredapi import Fred
from sklearn import linear_model

  
def get_sample_mean(data):
    return data.mean()

def get_sample_cov(data):
    return data.cov()

def get_sharpe_cov(data):
    
    '''
        The n-1 columns corresponds to the stock tickers
        The nth columns corresponds to the market index
    '''
    
    tickers = data.columns[:-1].tolist()
    mkt = data.columns[-1]
    mkt_var = np.var( data[mkt].values )
    
    regr = linear_model.LinearRegression()
    
    beta = np.zeros( data.shape[1] - 1 )
    res_var = np.zeros( data.shape[1] - 1 )
    
    for i, c in enumerate(data.columns[:-1]):
        regr.fit( data[mkt].values.reshape(-1, 1), data[c] )
        beta[i] = regr.coef_[0] 
        y_pred = regr.predict(data[mkt].values.reshape(-1, 1))
        res_var[i] = np.var( y_pred - data[c].values ) 

    return pd.DataFrame(mkt_var * np.dot( beta.reshape( -1, 1 ), beta.reshape( 1, -1 ) ) + np.diag( res_var ), 
                        index=tickers, columns=tickers)


def get_pi(data, sample_mean, sample_cov):
    mktIndex = data.columns[-1]
    
    x = data.drop( [mktIndex], axis=1 ).values
    m = sample_mean.drop( [mktIndex] ).values
    xm = x - m[np.newaxis,:]
    
    s = sample_cov.drop( [mktIndex] ).drop( [mktIndex], axis=1 ).values
    T = x.shape[0]
    
    t1 = np.dot( (xm**2).T, xm**2 )
    t2 = T * s**2
    t3 = s * np.dot( xm.T, xm )
    
    P = ( t1 + t2 - 2 * t3 ) / T
    return P

def get_rho(data, sample_mean, sample_cov, sharpe_cov, P):
    mktIndex = data.columns[-1]
    
    s0 = sample_cov[mktIndex].drop([mktIndex]).values
    s00 = sample_cov.loc[mktIndex, mktIndex]
    m0 = sample_mean[mktIndex]

    x = data.drop( [mktIndex], axis=1 ).values
    m = sample_mean.drop([mktIndex]).values
    s = sample_cov.drop([mktIndex]).drop([mktIndex], axis=1).values
    
    T = x.shape[0]
    
    x0m = data[mktIndex].values - m0
    xm = x - m[np.newaxis,:]
    
    t1 = np.dot( xm.T**2 , xm * x0m[:, np.newaxis]) * s0[np.newaxis, :] / s00
    t2 = np.dot( (xm * x0m[:, np.newaxis]).T , xm**2 ) * s0[:, np.newaxis] / s00
    t3 = np.dot( (xm * (x0m**2)[:, np.newaxis]).T , xm ) * np.dot( s0.reshape(-1,1), s0.reshape(1,-1) ) / s00**2
    t4 = T * sharpe_cov.values * s 
    
    R = ( t1 + t2 - t3 - t4 ) / T

    row, col = np.diag_indices_from(R)
    R[row, col] = P[row, col]
    
    return R

def get_gamma(mktIndex, sample, sharpe):
    return (sample.drop([mktIndex]).drop([mktIndex], axis=1).values - sharpe.values)**2

def get_linearLW_cov(data):
    tickers = data.columns[:-1].tolist()
    
    mktIndex = data.columns[-1]
    sample_mean = get_sample_mean(data)
    sample_cov = get_sample_cov(data)
    sharpe_cov = get_sharpe_cov(data)
    
    T = data.shape[0]
    
    Pi = get_pi(data, sample_mean, sample_cov)
    Rho = get_rho(data, sample_mean, sample_cov, sharpe_cov, Pi)
    Gamma = get_gamma(mktIndex, sample_cov, sharpe_cov)
    
    p = np.sum(Pi)
    r = np.sum(Rho)
    c = np.sum(Gamma)
    
    k = ( p - r ) / c
    
    alpha = k / T
    print('alpha: ', alpha)
    
    Cov = pd.DataFrame(( 1 - alpha ) * sample_cov.drop([mktIndex]).drop([mktIndex], axis=1) + alpha * sharpe_cov,
                       index=tickers, columns=tickers)
    return Cov

def get_clippingMP_cov(data):
    return 0

def get_RIE_cov(data):
    return 0

def get_equal_weight(tickers):
    ncol = len(tickers)
    weight = [1 / float(ncol) for i in range(ncol)]
    return dict(zip(tickers, weight))

def get_markowitz_weight(mktIndex, tickers, est_mean, est_cov, risk_level=1, short=False):  
    m = est_mean[tickers]
    s = est_cov[tickers].loc[tickers]
    
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
        weight = [1 if ticker == m.argmax() else 0 for ticker in tickers]
    elif risk_level == 'inf':
        weight = quadprog.solve_qp( s.values, 0*m.values, A, B, 1 )[0]
    else:
        weight = quadprog.solve_qp( risk_level*s.values, m.values, A, B, 1 )[0]
    return dict(zip(tickers, weight))


def backtest(data, weight, legend=None):
    newdata = data.copy()
    newdata['Portfolio'] = np.sum([weight[col] * newdata[col].values for col in weight.keys() ], axis=0)
    newdata['Return'] = newdata['Portfolio'].pct_change()
    newdata.iat[0, -1] = 0
    newdata['CompoundReturn'] = np.cumprod(1 + newdata['Return'])
    return newdata
   
if __name__ == "__main__":
    '''-------------------------------------------------------------------------'''

    '''                               Get data                                  '''

    '''-------------------------------------------------------------------------'''
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

    fred = Fred(api_key='cff534aff3dff5bdea6999ec25183677')
    mkt_data = fred.get_series('SP500', '1/1/2010', '1/1/2018')

    data3 = pd.concat([data3, mkt_data], axis=1, join_axes=[data3.index])
    mktIndex = 'SPX'
    data3.columns = data3.columns[:-1].tolist() + [mktIndex]

    data4 = data3.pct_change()
    data4 = data4.drop(data4.index[0], axis=0)

    insample_price = data3.iloc[:1500]
    outsample_price = data3.iloc[1500:]

    insample_ret = data4.iloc[:1500]
    outsample_ret = data4.iloc[1500:]



    '''-------------------------------------------------------------------------'''

    '''                             Optimization                                '''

    '''-------------------------------------------------------------------------'''


    est_mean = get_sample_mean(insample_ret)
    est_cov = get_sample_cov(insample_ret)
    sharpe_cov = get_sharpe_cov(insample_ret)
    linearLW_cov = get_linearLW_cov(insample_ret)


    eqw = get_equal_weight(tickers_to_keep)
    p_eqw = backtest(outsample_price, eqw)

    w1 = get_markowitz_weight(mktIndex, tickers_to_keep, est_mean, est_cov, risk_level=100)
    p1 = backtest(outsample_price, w1)

    w2 = get_markowitz_weight(mktIndex, tickers_to_keep, est_mean, est_cov, risk_level=0.01)
    p2 = backtest(outsample_price, w2)

    w3 = get_markowitz_weight(mktIndex, tickers_to_keep, est_mean, est_cov, risk_level=10000)
    p3 = backtest(outsample_price, w3)

    w4 = get_markowitz_weight(mktIndex, tickers_to_keep, est_mean, est_cov, risk_level=0)
    p4 = backtest(outsample_price, w4)

    w5 = get_markowitz_weight(mktIndex, tickers_to_keep, est_mean, est_cov, risk_level='inf')
    p5 = backtest(outsample_price, w5)

    w6 = get_markowitz_weight(mktIndex, tickers_to_keep, est_mean, sharpe_cov, risk_level='inf')
    p6 = backtest(outsample_price, w6)

    w7 = get_markowitz_weight(mktIndex, tickers_to_keep, est_mean, linearLW_cov, risk_level='inf')
    p7 = backtest(outsample_price, w7)


    fig, ax = plt.subplots(4,1)
    ax[0].bar(eqw.keys(), eqw.values(), label='EQW')
    ax[1].bar(w1.keys(), w1.values(), label='SAMPLE')
    ax[2].bar(w6.keys(), w6.values(), label='SHARPE')
    ax[3].bar(w7.keys(), w7.values(), label='LW')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    plt.show()

    plt.figure()
    plt.plot(p_eqw.index, p_eqw['CompoundReturn'], label='EQW')
    plt.plot(p1.index, p1['CompoundReturn'], label='SAMPLE')
    plt.plot(p6.index, p6['CompoundReturn'], label='SHARPE')
    plt.plot(p7.index, p7['CompoundReturn'], label='LW')
    plt.legend(loc=0)
    plt.grid()
    plt.show()
