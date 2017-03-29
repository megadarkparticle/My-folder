

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
import datetime




def get_data(file_name):
    df = pd.DataFrame()
    df = pd.read_csv((file_name), parse_dates=True, na_values=['nan'])
    df.head()
    return df

def get_ranking(column_name,df,parameter_name,ascending,div_20):
    
    if div_20 == True:
        df.loc[:,column_name] = (df.groupby('Date')[parameter_name].rank(method='dense',ascending=ascending))/20
    else :
        df.loc[:,column_name] = (df.groupby('Date')[parameter_name].rank(method='dense',ascending=ascending))/df.groupby('Date')['Date'].transform('count')
    return df

def regression(Y,df):
    X = df.drop(Y,1)
    result = sm.ols(formula='df[Y] ~ X',data=df).fit()
    return result.summary()

def shift_arbitrage_index(df):
    
    df.loc[:,'Arb1'] = df.Arbitrage_risk_mispricing_index.shift(1)
    df.loc[:,'Arb2'] = df.Arbitrage_risk_mispricing_index.shift(2) 
    df.loc[:,'Arb3'] = df.Arbitrage_risk_mispricing_index.shift(3)
    df.loc[:,'Arb4'] = df.Arbitrage_risk_mispricing_index.shift(4) 
    df.loc[:,'Arb5'] = df.Arbitrage_risk_mispricing_index.shift(5)
    df.loc[:,'Arb6'] = df.Arbitrage_risk_mispricing_index.shift(6) 
    df.loc[:,'Arb7'] = df.Arbitrage_risk_mispricing_index.shift(7)
    df.loc[:,'Arb8'] = df.Arbitrage_risk_mispricing_index.shift(8) 
    df.loc[:,'Arb9'] = df.Arbitrage_risk_mispricing_index.shift(9)
    df.loc[:,'Arb10'] = df.Arbitrage_risk_mispricing_index.shift(10)
    df.loc[:,'Arb11'] = df.Arbitrage_risk_mispricing_index.shift(11) 
    df.loc[:,'Arb12'] = df.Arbitrage_risk_mispricing_index.shift(12)
    df.loc[:,'Arb13'] = df.Arbitrage_risk_mispricing_index.shift(13) 
    df.loc[:,'Arb14'] = df.Arbitrage_risk_mispricing_index.shift(14)
    df.loc[:,'Arb15'] = df.Arbitrage_risk_mispricing_index.shift(15) 
    df.loc[:,'Arb16'] = df.Arbitrage_risk_mispricing_index.shift(16)
    df.loc[:,'Arb17'] = df.Arbitrage_risk_mispricing_index.shift(17) 
    df.loc[:,'Arb18'] = df.Arbitrage_risk_mispricing_index.shift(18)
    df.loc[:,'Arb19'] = df.Arbitrage_risk_mispricing_index.shift(19)
    df.ix[:,'Arb0_9'] = df.ix[:,[41,42,43,44,45,46,47,48,49,50]].mean(axis=1)
    df.ix[:,'Arb0_19'] = df.ix[:,[41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]].mean(axis=1)
    df = df.drop(df.columns[[42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]], axis=1)

    return df

def shift_lottery_index(df):
    
    df.loc[:,'lot1'] = df.Lottery_like_mispricing_index.shift(1)
    df.loc[:,'lot2'] = df.Lottery_like_mispricing_index.shift(2) 
    df.loc[:,'lot3'] = df.Lottery_like_mispricing_index.shift(3)
    df.loc[:,'lot4'] = df.Lottery_like_mispricing_index.shift(4) 
    df.loc[:,'lot5'] = df.Lottery_like_mispricing_index.shift(5)
    df.loc[:,'lot6'] = df.Lottery_like_mispricing_index.shift(6) 
    df.loc[:,'lot7'] = df.Lottery_like_mispricing_index.shift(7)
    df.loc[:,'lot8'] = df.Lottery_like_mispricing_index.shift(8) 
    df.loc[:,'lot9'] = df.Lottery_like_mispricing_index.shift(9)
    df.loc[:,'lot10'] = df.Lottery_like_mispricing_index.shift(10)
    df.loc[:,'lot11'] = df.Lottery_like_mispricing_index.shift(11) 
    df.loc[:,'lot12'] = df.Lottery_like_mispricing_index.shift(12)
    df.loc[:,'lot13'] = df.Lottery_like_mispricing_index.shift(13) 
    df.loc[:,'lot14'] = df.Lottery_like_mispricing_index.shift(14)
    df.loc[:,'lot15'] = df.Lottery_like_mispricing_index.shift(15) 
    df.loc[:,'lot16'] = df.Lottery_like_mispricing_index.shift(16)
    df.loc[:,'lot17'] = df.Lottery_like_mispricing_index.shift(17) 
    df.loc[:,'lot18'] = df.Lottery_like_mispricing_index.shift(18)
    df.loc[:,'lot19'] = df.Lottery_like_mispricing_index.shift(19)
    df.ix[:,'lot0_9'] = df.ix[:,[44,45,46,47,48,49,50,51,52,53]].mean(axis=1)
    df.ix[:,'lot0_19'] = df.ix[:,[44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]].mean(axis=1)
    df = df.drop(df.columns[[45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]], axis=1)

    return df

def shift_diff_Arb_index(df):
    
    df.loc[:,'diff_Arb1'] = df.diff_Arb_index.shift(1)
    df.loc[:,'diff_Arb2'] = df.diff_Arb_index.shift(2) 
    df.loc[:,'diff_Arb3'] = df.diff_Arb_index.shift(3)
    df.loc[:,'diff_Arb4'] = df.diff_Arb_index.shift(4) 
    df.loc[:,'diff_Arb5'] = df.diff_Arb_index.shift(5)
    df.loc[:,'diff_Arb6'] = df.diff_Arb_index.shift(6) 
    df.loc[:,'diff_Arb7'] = df.diff_Arb_index.shift(7)
    df.loc[:,'diff_Arb8'] = df.diff_Arb_index.shift(8) 
    df.loc[:,'diff_Arb9'] = df.diff_Arb_index.shift(9)
    df.loc[:,'diff_Arb10'] = df.diff_Arb_index.shift(10)
    df.loc[:,'diff_Arb11'] = df.diff_Arb_index.shift(11) 
    df.loc[:,'diff_Arb12'] = df.diff_Arb_index.shift(12)
    df.loc[:,'diff_Arb13'] = df.diff_Arb_index.shift(13) 
    df.loc[:,'diff_Arb14'] = df.diff_Arb_index.shift(14)
    df.loc[:,'diff_Arb15'] = df.diff_Arb_index.shift(15) 
    df.loc[:,'diff_Arb16'] = df.diff_Arb_index.shift(16)
    df.loc[:,'diff_Arb17'] = df.diff_Arb_index.shift(17) 
    df.loc[:,'diff_Arb18'] = df.diff_Arb_index.shift(18)
    df.loc[:,'diff_Arb19'] = df.diff_Arb_index.shift(19)
    df.ix[:,'Cul_diff_Arb0_9'] = df.ix[:,[47,48,49,50,51,52,53,54,55,56]].sum(axis=1)
    df.ix[:,'Cul_diff_Arb0_19'] = df.ix[:,[47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66]].sum(axis=1)
    df=df.drop(df.columns[[48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66]], axis=1)
    return df
    

def test_run():
    # Read data
    df = get_data("Bot20v5.csv")
    
    get_ranking('Ranking_by_Size',df,'Size_2',True,True) #small size is low rank #29
    get_ranking('Ranking_by_BTM',df,'BTM_2',False,False) #high BTM is low rank #30
    get_ranking('Ranking_by_MOM',df,'Mom_240__21',False,True) #high MOM is low rank  #31
    get_ranking('Ranking_by_REV',df,'Rev_20__2',True,True) #low REV is low rank #32
    get_ranking('Ranking_by_ASSGR',df,'ASSGR_2',True,False) #low ASSGR is low rank #33
    get_ranking('Ranking_by_ROA',df,'ROA_2',False,True) #high ROA is low rank #34
    get_ranking('Ranking_by_CEQIS',df,'CEQ_240__2',True,True) #low CEQIS is low rank #35
    get_ranking('Ranking_by_ILLIQ',df,'ILLIQ_20__2',False,True) #high ILLIQ is low #36
    get_ranking('Ranking_by_ISKEW',df,'ISKEW_20__2',True,True) #low ISKEW[-20,-2] is low rank #37
    get_ranking('Ranking_by_MAX',df,'Max_20__2',True,True) #low MAX is low rank #38
    get_ranking('Ranking_by_REVxRELVLM',df,'REVxRELVLM_20__2',True,False) #low REVxRELVLM is low rank #39
    get_ranking('Ranking_by_PRICE',df,'PRICE_20__2',False,False) #High PRICE-20,-2 is low rank #40
    get_ranking('Ranking_by_IVOL',df,'IVOL_20__2',True,False) #Low IVOL-20,-2 is low rank #41
               
    
    df.loc[:,'Arbitrage_risk_mispricing_index'] = df.ix[:,[31,32,33,34,35,36,39]].mean(axis=1) #42
    df=shift_arbitrage_index(df)
    df.loc[:,'Lottery_like_mispricing_index'] = df.ix[:,[37,38,40,41]].mean(axis=1) 
    df = shift_lottery_index(df)
    
    df.loc[:,'diff_Arb_index'] = (df.Arbitrage_risk_mispricing_index[1:]/df.Arbitrage_risk_mispricing_index[:-1].values)-1
    df.diff_Arb_index[0]=0
    df=shift_diff_Arb_index(df)

    df.loc[:,'diff_Arb_index_shift20'] = df.diff_Arb_index.shift(20)
    df.loc[:,'Cul_diff_Arb0_9_shift20'] = df.Cul_diff_Arb0_9.shift(20)
    df.loc[:,'Cul_diff_Arb0_19_shift20'] = df.Cul_diff_Arb0_19.shift(20)
    
    #10days, 20days data
    df.loc[:,'A'] = (df['Date'].rank(method='min',ascending=True))
    df.loc[:,'B'] = (df.A.rank(method='dense',ascending=True))
    df10 = df[df.B%10==1]
    df20 = df[df.B%20==1]
    df.drop(['A','B'], axis=1, inplace=True)
    df10.drop(['A','B'], axis=1, inplace=True)
    df20.drop(['A','B'], axis=1, inplace=True)
    
    #df.corr()

    """///////////////////////////////////////////////////////////////////////////////////////////    
     'R_Rf', 'Rm_Rf', 'AR_b20_2', 'AR_b40_21', 'AR_b40_2', 'AR_b240_2', 'AR0_9', 'AR0_19', 'AR20_39',
     'Size_2', 'BTM_2','Mom_240__21', 'Rev_20__2', 'ASSGR_2', 'ROA_2', 'CEQ_240__2', 'ILLIQ',
     'ILLIQ_20__2', 'RELVLM', 'RELVLM_20__2', 'REVxRELVLM_20__2', 'Max_20__2', 'IVOL_20__2',  'IVOL_40__2', 
     'ISKEW_20__2', 'PRICE_20__2', 'Ranking_by_Size', 'Ranking_by_BTM', 'Ranking_by_MOM',
     'Ranking_by_REV', 'Ranking_by_ASSGR', 'Ranking_by_ROA', 'Ranking_by_CEQIS', 'Ranking_by_ILLIQ',
     'Ranking_by_ISKEW', 'Ranking_by_MAX', 'Ranking_by_REVxRELVLM', 'Ranking_by_PRICE',
     'Ranking_by_IVOL', 'Arbitrage_risk_mispricing_index', 'Lottery_like_mispricing_index'
     ,'Arb0_9','Arb0_19','lot0_9','lot0_19', 'diff_Arb_index','Cul_diff_Arb0_9','Cul_diff_Arb0_19',
     'diff_Arb_index_shift20','Cul_diff_Arb0_9_shift20','Cul_diff_Arb0_19_shift20']
    #/////////////////////////////////////////////////////////////////////////////////////"""
    #Choose the data for regression
    
    #df1 = df.iloc[:,[7,11,12,13,14,15,16,19,22,27,42]]
    #regression('AR_b240_2', df1)

    #df2 = df.iloc[:,[7,11,12,13,14,15,16,19,22,27,42]]
   # regression('AR_b240_2', df2)
    
    result = sm.ols(formula='AR_b240_2 ~ Arbitrage_risk_mispricing_index',data=df).fit()
    result.summary()
    
    result = sm.ols(formula=' AR0_9 ~ Arb0_9',data=df10).fit()
    result.summary()
    
    result = sm.ols(formula=' AR0_19 ~ Arb0_19',data=df20).fit()
    result.summary()

    result = sm.ols(formula='AR_b240_2 ~ Size_2 +BTM_2 +Mom_240__21+ Rev_20__2+ASSGR_2+ROA_2+CEQ_240__2+ ILLIQ_20__2+REVxRELVLM_20__2+PRICE_20__2+ Arbitrage_risk_mispricing_index',data=df).fit()
    result.summary()
    
    result = sm.ols(formula=' AR0_19 ~ Size_2 +BTM_2 +Mom_240__21+ Rev_20__2+ASSGR_2+ROA_2+CEQ_240__2+ ILLIQ_20__2+REVxRELVLM_20__2+PRICE_20__2+ Arb0_19',data=df20).fit()
    result.summary()
    
    result = sm.ols(formula=' AR0_9 ~ Size_2 +BTM_2 +Mom_240__21+ Rev_20__2+ASSGR_2+ROA_2+CEQ_240__2+ ILLIQ_20__2+REVxRELVLM_20__2+PRICE_20__2+ Arb0_9',data=df10).fit()
    result.summary()
    
    result = sm.ols(formula='AR_b240_2 ~ Lottery_like_mispricing_index',data=df).fit()
    result.summary()
    
    result = sm.ols(formula=' AR0_9 ~ lot0_9',data=df10).fit()
    result.summary()
    
    result = sm.ols(formula=' AR0_19 ~ lot0_19',data=df20).fit()
    result.summary()
    
    result = sm.ols(formula='AR_b240_2 ~ Size_2+ BTM_2+ Max_20__2+IVOL_20__2+ISKEW_20__2+PRICE_20__2+Lottery_like_mispricing_index',data=df).fit()
    result.summary()

    result = sm.ols(formula='AR0_19 ~ Size_2+ BTM_2+ Max_20__2+IVOL_20__2+ISKEW_20__2+PRICE_20__2+lot0_19',data=df20).fit()
    result.summary()

    result = sm.ols(formula='AR0_9 ~ Size_2+ BTM_2+ Max_20__2+IVOL_20__2+ISKEW_20__2+PRICE_20__2+lot0_9',data=df10).fit()
    result.summary()

    #////////////////////////////////////////////////////////////////////////////////////////////
    result = sm.ols(formula='diff_Arb_index ~ IVOL_20__2 + Lottery_like_mispricing_index',data=df).fit()
    result.summary()
    
    result = sm.ols(formula='Cul_diff_Arb0_9 ~ IVOL_20__2 + lot0_9',data=df10).fit()
    result.summary()
    
    result = sm.ols(formula='Cul_diff_Arb0_19 ~ IVOL_20__2 + lot0_19',data=df20).fit()
    result.summary()

    result = sm.ols(formula='diff_Arb_index ~ IVOL_40__2 + Lottery_like_mispricing_index',data=df).fit()
    result.summary()

    result = sm.ols(formula='Cul_diff_Arb0_9 ~ IVOL_40__2 + lot0_9',data=df10).fit()
    result.summary()

    result = sm.ols(formula='Cul_diff_Arb0_19 ~ IVOL_40__2 + lot0_19',data=df20).fit()
    result.summary()
    #//////////////////////////////////
    result = sm.ols(formula='diff_Arb_index_shift20 ~ IVOL_20__2 + Lottery_like_mispricing_index',data=df).fit()
    result.summary()
    
    result = sm.ols(formula='Cul_diff_Arb0_9_shift20 ~ IVOL_20__2 + lot0_9',data=df10).fit()
    result.summary()
    
    result = sm.ols(formula='Cul_diff_Arb0_19_shift20 ~ IVOL_20__2 + lot0_19',data=df20).fit()
    result.summary()

    result = sm.ols(formula='diff_Arb_index_shift20 ~ IVOL_40__2 + Lottery_like_mispricing_index',data=df).fit()
    result.summary()

    result = sm.ols(formula='Cul_diff_Arb0_9_shift20 ~ IVOL_40__2 + lot0_9',data=df10).fit()
    result.summary()

    result = sm.ols(formula='Cul_diff_Arb0_19_shift20 ~ IVOL_40__2 + lot0_19',data=df20).fit()
    result.summary()









if __name__ == "__main__":
    test_run()
