
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    #if 'SPY' not in symbols:  # add SPY for reference, if absent
        #symbols.insert(0, 'SPY')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        df = df[np.isfinite(df['AAC.AX'])] # drop the row when nan exist in ALL.AX's row

    return df


def symbol_to_path(symbol, base_dir="data_for_portfolio_management"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    # TODO: Your code here
    # Note: Returned DataFrame must have the same number of rows
    daily_returns = df.copy() # copy data frame to size and column
    
    daily_returns[1:] = np.log(daily_returns[1:]/daily_returns[:-1].values)
    daily_returns.ix[:,'Z'] = ["a"]
    daily_returns.Z[0] ="b" 
    daily_returns=daily_returns[daily_returns.Z!="b"]
    daily_returns = daily_returns.drop('Z', axis=1)

    return daily_returns
    
def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def allocation():
    
    """alloc=[0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04]"""
    alloc= np.array([0.0000	,0.0000	,0.4371,	0.3025,	0.0000,	0.0259,	0.0000,
    0000	,0.0000	,0.0000	,0.0000	,0.0110	,0.0000	,0.0000	,
    0.0000	,0.0000	,0.0000,	0.0000	,0.0584,	0.0000	,0.0000
    ,0000	,0.1070	,0.0000	,0.0581])
    
             
    return alloc

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return np.asmatrix(k / sum(k))


def annual_return(weight, mean_ret):
    weight_return = weight*mean_ret.T

    return 12*weight_return

def annual_portfolio_std(cov,weight):
    weight_cov = cov.dot(weight)
    std = weight_cov.sum()
    
    return std

def calc_portfolio_var(returns, weights =None):
    if (weights is None):
        weights = np.ones(returns.columns.size)
        returns.columns.size
    Var_Cov_martrix = np.cov(daily_returns.T,ddof=1)
    var = Weight*Var_Cov_martrix*Weight.T
    return var
    
def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty

    return mu, sigma


def solve_weights(R, C, rf):
    def fitness(W, R, C, rf):
        # calculate mean/variance of the portfolio
        mean, var = port_mean_var(W, R, C)  
        util = (mean - rf) / sqrt(var)      # utility = Sharpe ratio
        return 1/util                       # maximize the utility
    n = len(R)
    W = ones([n])/n                     # start with equal weights
    b_ = [(0.,1.) for i in range(n)]    # weights between 0%..100%. 
                                        # No leverage, no shorting
    c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })   # Sum of weights = 100%
    optimized = scipy.optimize.minimize(fitness, W, (R, C, rf), 
                method='SLSQP', constraints=c_, bounds=b_)  
    if not optimized.success: 
        raise BaseException(optimized.message)
    return optimized.x  # Return optimized weights



def test_run():
    # Read data
    dates = pd.date_range('2014-03-03', '2017-03-01') 
    symbols = ['AAC.AX','ABC.AX','AGL.AX','ALL.AX','APA.AX','API.AX','ASX.AX'
           ,'BHP.AX','CBA.AX','CPU.AX','CSL.AX','FMG.AX','IAG.AX','LLC.AX'
           ,'ORG.AX','ORI.AX','PPT.AX','SEK.AX','SPK.AX','SUN.AX','SYD.AX'
           ,'TLS.AX','WEB.AX','WES.AX','WFD.AX']

    number_of_stocks=len(symbols)
    df = get_data(symbols, dates)
    daily_returns = compute_daily_returns(df)
    #plot_data(daily_returns, title="Monthly returns", ylabel="Daily returns")
    Weight = rand_weights(number_of_stocks)
    #Weight = np.asmatrix(allocation())
    
    mean_ret = np.asmatrix(daily_returns[1:].mean(axis =0)) # don't average first row
    return_annulise= annual_return(Weight,mean_ret)
    Var_port_monthly = calc_portfolio_var(daily_returns, Weight)   
    Sd = np.sqrt(Var_port_monthly)
    Sd_annualise = np.sqrt((Sd**2)*12)
    Rf = 0.015
    Sharp_ratio = (return_annulise - Rf)/Sd_annualise
    
    
    
                  
    k = np.random.rand(25)             
    #Optimiser
    """n_portfolios = 10000
    means, stds = np.column_stack([random_portfolio(daily_returns) 
    for _ in range(n_portfolios)])
    
    plt.plot(stds, means, 'o', markersize=5)
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.title('Mean and standard deviation of returns of randomly generated portfolios')"""

 
if __name__ == "__main__":
    test_run()
