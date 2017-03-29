
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


def annual_return(weight, mean_ret):
    weight_return = weight*mean_ret
    sum_weight_return = weight_return.sum()
    
    return 12*sum_weight_return

def annual_portfolio_std(cov,weight):
    weight_cov = cov.dot(weight)
    std = weight_cov.sum()
    
    return std

def calc_portfolio_var(returns, weights =None):
    if (weights is None):
        weights = np.ones(returns.columns.size)
        returns.columns.size
    sigma = np.cov(returns.T,ddof=0)
    var = (weights*sigma*weights.T).sum()
    return var
    
def test_run():
    # Read data
    dates = pd.date_range('2014-03-03', '2017-03-01') 
    symbols = ['AAC.AX','ABC.AX','AGL.AX','ALL.AX','APA.AX','API.AX','ASX.AX'
           ,'BHP.AX','CBA.AX','CPU.AX','CSL.AX','FMG.AX','IAG.AX','LLC.AX'
           ,'ORG.AX','ORI.AX','PPT.AX','SEK.AX','SPK.AX','SUN.AX','SYD.AX'
           ,'TLS.AX','WEB.AX','WES.AX','WFD.AX']
    
    """ symbols = ['ALL.AX','WEB.AX','WES.AX','AAC.AX','ORG.AX','CBA.AX','SUN.AX'
           ,'ASX.AX','IAG.AX','PPT.AX','CSL.AX','API.AX','SYD.AX','SEK.AX'
           ,'CPU.AX','BHP.AX','FMG.AX','ORI.AX','ABC.AX','WFD.AX','LLC.AX'
           ,'TLS.AX','SPK.AX','AGL.AX','APA.AX']"""
    
                  
    df = get_data(symbols, dates)
    plot_data(df)

    # Compute daily returns
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title="Monthly returns", ylabel="Daily returns")
    Weight = allocation()

    mean_ret = daily_returns[1:].mean(axis =0) # don't average first row
    a=annual_return(Weight,mean_ret)
    print(a)
    covariance = daily_returns.cov()
    sigma = np.cov(daily_returns.T,ddof=1)
    """
    d=Weight.transpose()
    *covariance*Weight
 
    b = annual_portfolio_std(covariance,Weight)"""
    c=Weight.transpose()*covariance*Weight
    e = calc_portfolio_var(daily_returns, Weight)   
    f = np.sqrt(e)
    Var = Weight*sigma*Weight.T    
if __name__ == "__main__":
    test_run()
