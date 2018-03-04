# -*- coding: utf-8 -*-

import numpy 
import sklearn
import pandas as pd
import os
import datetime
from sklearn.decomposition import *
import matplotlib.pyplot as plt

#del data_df
#del data

os.chdir('C:\Users\Clement Natta\Desktop\PFE')

with open('Tickers.txt') as tickers:
    reader=tickers.read().split("\n")
    list_tickers=[read for read in reader]

#data=pd.read_csv('AAPL.txt'.format(ticker), sep=",")
data=pd.read_csv('AAPL.txt', sep=",")
data['DATE']=data['DATE'].apply(lambda x : str(x))
data['DATE']=data['DATE'].apply(lambda x : datetime.datetime.strptime(x,'%Y%m%d'))

data.index=data["DATE"]


data=data[" OPEN"]
data_df=pd.DataFrame(data)

print(type(data_df))
damaged_stocks = []
nondamaged_stocks = ['AAPL']
for ticker in list_tickers[2:]:        
    df = pd.read_csv('{}.txt'.format(ticker.strip()), sep=",")
    df['DATE']=df['DATE'].apply(lambda x : str(x))
    df['DATE']=df['DATE'].apply(lambda x : datetime.datetime.strptime(x,'%Y%m%d'))
    df.index=df["DATE"]
    df = pd.DataFrame(df[" OPEN"])
    if len(df)<3000:
        #print(ticker)
        damaged_stocks.append(ticker.strip())        
    else :
        #print(type(df))
        data_df=data_df.merge(df,left_index=True,right_index=True)
        nondamaged_stocks.append(ticker.strip())  
    #data=data.merge(data,df)
    
#data_df.columns=list_tickers[1:]
data_df.columns=nondamaged_stocks
data_df=data_df.transpose()
# replicate Data from question in DataFrame


def scatterplot(x_data, y_data, x_label, y_label, title):
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, s = 30, color = '#539caf', alpha = 0.75)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.autofmt_xdate()

#use column headers as x values
x = data_df.columns
# sum all values from DataFrame along vertical axis
y = data_df.loc['AAPL']
dates=data_df.columns

lag=[(dates[i]-dates[i+1]).days for i in range (len(dates)-1)]
lag_dataframe=pd.DataFrame(lag)
lag_dataframe.plot(kind='hist')
plt.show()
    
