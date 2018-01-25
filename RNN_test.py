#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:39:05 2017

@author: Drogon

Please modify the data_path string below to the directory where 'RawData.csv' is stored
Egs: data_path = '/Users/AllData/'

Note: folder_name string below should be the name of the folder_
(inside the data_path folder) in which the parameters were exported in the_
learning phase

"""

data_path = '/Users/Drogon/Documents/Columbia/Courses/Fall_2017/Big_Data_Choromanski/data/data'
folder_name = 'dummy' ############# CHANGE THIS

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time as tm

#%%
os.chdir(data_path)

data=pd.read_csv('RawData.csv', sep=",")


#%%
df = pd.DataFrame(data['Volume'])
df.index = data['Ntime']

data1 = data.copy()
del data1['Ntime']
data1.drop(['time', 'Close Price','Volume','Low Price', 'High Price'], axis=1, inplace=True)


#%%
#Initializing the dataset 

data_df=data1['Open Price'].copy()

#%%
return_df = (data_df - data_df.shift(1))/data_df
return_df = pd.DataFrame(return_df)
returns = return_df.values.tolist()

#%%
train_num = 1500 + 25
num_unrollings =  1
look_back = 20 
delta=2e-4 #transaction cost (bps)

#%%


def learning_rate_exponential(rate0, gamma, global_, decay):
    return rate0*gamma**(global_/decay)


def np_sigmoid(x):
    return 1/(1+np.exp(-x))
def np_relu_modified(x):
    return (np.exp(x)-1)

def np_sigmoid_modified(w,center,b_):
    return (np.exp(b_*((w-center)/center))/(1+np.exp(b_*(w-center)/center))-center)/(1/(1+np.exp(-b_))-center)

def np_tanh(x):
  #return (tf.exp(x)-tf.exp(-x))/(tf.exp(x)+tf.exp(-x))
   return 2*np_sigmoid(2*x)-1 

time1 = tm.time()

#%%

def compute_look_back(returns, train_num, look_back, test_i):
    return returns[train_num + test_i - look_back : train_num + test_i].copy()


def run_layer(U,W,b,V,c, look_back_data, F_prev):
    ''' phi2(phi1((x*U, o_*W) + b)*V + c)  '''
    x = np.array(look_back_data).reshape(1,len(look_back_data))
    #print(x, 'type of x= ', type(x), x.shape)
    o_ = np.array(F_prev).reshape(1,1)
    a_ = np.concatenate((np.matmul(x,U), np.matmul(o_, W)), axis = 1)
    h_output = np_tanh(a_ + np.array(b))
    return np_tanh(np.matmul(h_output, V) + np.array(c))


###
num_updates = 150



model_return = []
model_signal = []
market_signal = []
market_return = []
cum_market_return = []

test_i = 0
  
results_path = data_path + folder_name + '/'

U_learned = pd.read_csv(results_path + 'U.csv').values[:,1:]
W_learned = pd.read_csv(results_path + 'W.csv').values[:,1:]
V_learned = pd.read_csv(results_path + 'V.csv').values[:,1:]
b_learned = pd.read_csv(results_path + 'b.csv').values[:,1:]
c_learned = pd.read_csv(results_path + 'c.csv').values[:,1:]

for update_i in range(num_updates):
    
    for unrolling_i in range(num_unrollings):
        
        look_back_data = compute_look_back(returns, train_num, look_back, test_i)
        
        if (update_i == 0):
            F_prev = np.sign(look_back_data[-1])
        
        F_prev_ = F_prev
        
        F_prev = run_layer(U_learned, W_learned, b_learned, V_learned, c_learned, look_back_data, F_prev)
        
        model_signal.append(F_prev)
        realized_return = returns[train_num + test_i]
        market_signal.append(np.sign(realized_return))
        market_return.append(realized_return)
        temp_model_return = (F_prev_)*(realized_return) - delta*np.abs(F_prev_ - F_prev)
        
        model_return.append(temp_model_return)
        
        test_i += 1
    
#%%     
market_return = np.array(market_return).flatten().tolist()
plt.plot(market_return)

print(np.sum(market_return))

#%%     
model_return = np.array(model_return).flatten().tolist()
plt.plot(model_return)

print(np.sum(model_return))

#%%
comp_model_return = [1.]
for i in range(len(model_return)):
    comp_model_return.append(comp_model_return[-1]*(1+model_return[i]))
plt.plot(comp_model_return)

#%%
comp_market_return = [1.]
for i in range(len(market_return)):
    comp_market_return.append(comp_market_return[-1]*(1 + market_return[i]))
plt.plot(comp_market_return)

#%%
plt.plot(comp_model_return)
plt.plot(comp_market_return)
plt.legend(['model','market'])
