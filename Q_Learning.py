
import numpy as np
import sklearn
import pandas as pd
import os
import datetime
from sklearn.decomposition import *
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA
import random
from math import *

os.chdir('C:\\Users\\Clement Natta\\Desktop\\PFE')
    
#from optimization import *

import statsmodels
from scipy.stats import bernoulli

import pickle

        


##### Q-learning
        
def initLocalizeState(component):
    global granularity
    global Ind_components
    global k_it
    range_test=max(Ind_components[component])-min(Ind_components[component])
    number_state=int(range_test*granularity/np.std(Ind_components[component]))
    division=[np.std(Ind_components[component])*i/granularity for i in range(number_state+1)]
    return division

def LocalizeState(component_return,component,sub_division):
    global k_it
    bin_num=np.searchsorted(sub_division[component],component_return)
    bin_num=-1/2*(bin_num+len(sub_division[component]))*(k_it[component]-1)+bin_num*(k_it[component]+1)/2
    
    return int(bin_num)

    
def Q_update(state,action, component,time,nu):
    '''
       This function is used to implement the updates of the value functions 
    '''
    
    
    global Ind_components
    global Q
    global gamma
    global k_it
    global delta
    global demixing
    
    
    #print(sum(demixing[component,:]),k_it[component]-action)
    index_component=list(Ind_components.columns).index(component)
    #cost=log(1-delta*sum(demixing.iloc[index_component])*abs(k_it[component]-(-1*(1-action)+action)))
    cost=log(1-delta*abs(k_it[component]-(-1*(1-action)+action)))
    index=Ind_components.index[time+1]
    return_component=Ind_components[component][index]
    reward=log(1+(-1*(1-action)+action)*return_component)+cost
    next_state=LocalizeState(reward,component,subdivision)
    update=(1-nu)*Q[component][action,state]+nu*(reward+gamma*(max(Q[component][:,next_state])))
    
    Q[component][action,state]=update
        
    return Q[component][action,state]


def Q_update_error(state,action, component,time,nu,error):
    '''
       This function is used to implement the updates of the value functions 
    '''

    global Ind_components
    global Q
    global gamma
    global k_it
    global delta
    global demixing
    
    #print(sum(demixing[component,:]),k_it[component]-action)
    index_component=list(Ind_components.columns).index(component)
    #cost=log(1-delta*sum(demixing.iloc[index_component])*abs(k_it[component]-(-1*(1-action)+action)))
    cost=log(1-delta*abs(k_it[component]-(-1*(1-action)+action)))
    
    reward=log(1+(-1*(1-action)+action)*Ind_components[component][Ind_components.index[time+1]])+cost
    next_state=LocalizeState(reward,component,subdivision)
    update=(1-nu)*Q[component][action,state]+nu*(reward+gamma*(max(Q[component][:,next_state])))
    if abs(abs(update-Q[component][action,state])/Q[component][action,state])>error:
        Q[component][action,state]=update
        return Q[component][action,state] , 0
    else :
        return Q[component][action,state] , 1
    


def Q_lambda_update_error(state,action, component,time,nu, error):
    '''
       This function is used to implement the updates of the value functions 
    '''
    
    
    global Ind_components
    global Q
    global eligibility
    global gamma
    global k_it
    global delta
    global demixing
    global subdivision
    global lambda_
    '''
    The next part is commented because its not used anymore but can be useful in the future 
    as a first aproach to the value function with consists of using the mean return as the 
    approximation of the next return 
    '''
    
    #print(sum(demixing[component,:]),k_it[component]-action)
    index_component=list(Ind_components.columns).index(component)
    #cost=log(1-delta*sum(demixing.iloc[index_component])*abs(k_it[component]-(-1*(1-action)+action)))
    cost=log(1-delta*abs(k_it[component]-(-1*(1-action)+action)))
    
    reward=log(1+(-1*(1-action)+action)*Ind_components[component][Ind_components.index[time+1]])+cost
    next_state=LocalizeState(reward,component,subdivision)
    change=(reward+gamma*(max(Q[component][:,next_state])))-Q[component][action,state]
    
    if abs(abs(change)/Q[component][action,state])>error:

        eligibility[component][action,state]+=1
        Q[component]= Q[component]+change*nu*eligibility[component]          
        eligibility[component]=gamma*lambda_*eligibility[component]
             
        return Q[component][action,state] , 0
    else :
        return Q[component][action,state] , 1
    
    

def show_traverse(component):
    '''
       This function highlights the different transitions in the state space the idea
       was to use it to make sure that we have enough exploration
    '''
    
    global Q
    
    # show all the transitions
    for i in range(len(Q)):
        current_state = i
        traverse = "%i -> " % current_state
        n_steps = 0
        while current_state != 5 and n_steps < 20:
            next_action = np.argmax(Q[state,:,component])
            next_state=state+1
            current_state = next_state
            traverse += "%i -> " % current_state
            n_steps = n_steps + 1
        # cut off final arrow
        traverse = traverse[:-4]
        print("Greedy traversal for starting state %i" % i)
        print(traverse)
        print("")


#This definition conducts online policy learning it is more of a SARSA using an epsilon greedy technique but it should converge faster 

def Q_train_error(num_iterations,train,epsilon,sub_division,error):
    global Ind_components
    global Q
    global k_it
    global nu
    total_iteration=0
    for component in  Ind_components.columns:     
        convergence=0
        print('---------------   ******************  ----------------')
        print ('{}'.format(component)+'th component out of {}'.format(len(Ind_components.columns)))
        print('---------------   ******************  ----------------')
        print ('{}'.format(total_iteration+1) +'th iteration out of {}'.format(num_iterations))
        
        
        for iteration in range(num_iterations):  
            print('---------------   ******************  ----------------')
            print ('{}'.format(iteration+1) +'th iteration out of {}'.format(num_iterations))
         
            if convergence > 5*np.size(Q[component])/2:
                total_iteration=iteration
                break
            return_component=Ind_components[component][Ind_components.index[iteration]]
            for t in range(train):
                state=LocalizeState(return_component,component,sub_division)
                if random.uniform(0,1)<epsilon[iteration]:
                    
                    action=np.argmax(Q[component][:,state])
        
                else:
                    rand=random.randint(0,1)
                    action=rand
                    
                Q[component][action,state]=Q_lambda_update_error(state,action, component,t,nu[iteration,t],error)[0]
                convergence+=Q_lambda_update_error(state,action, component,t,nu[iteration,t],error)[1]
                k_it[component]=-1*(1-action)+action
            
    return Q

def Q_train_component_error(num_iterations,train,epsilon,sub_division,error,component):
    global Ind_components
    global Q
    global k_it
    global nu
    total_iteration=0
    convergence=0
    print('---------------   ******************  ----------------')
    precision=[]        
    for iteration in range(num_iterations):  
        print ('{}'.format(iteration+1) +'th iteration out of {}'.format(num_iterations))
        if convergence > 5*np.size(Q[component])/2:
            total_iteration=iteration
            break
      
        for t in range(train):
            return_component=Ind_components[component][Ind_components.index[t]]
            state=LocalizeState(return_component,component,sub_division)
            if random.uniform(0,1)<epsilon[iteration]:
                
                action=np.argmax(Q[component][:,state])
    
            else:
                rand=random.randint(0,1)
                action=rand
                
            #Q[component][action,state]=Q_update_error(state,action, component,t,nu[iteration,t],error)[0]
            
            Q[component][action,state]=1
            convergence+=Q_update_error(state,action, component,t,nu[iteration,t],error)[1]
            k_it[component]=-1*(1-action)+action
        precision.append(test_signal_component(0, train,component))
    
    return Q, precision[len(precision)-1]

def test_signal_component(start_date, end_date,component):
    
    global train_size
    global Ind_components
    global subdivision
    
    signal_convergence=[]
    for t in range(start_date, end_date):
        return_component=Ind_components[component][Ind_components.index[t]]
        state=LocalizeState(return_component,component,subdivision)
        action=-1*(1-np.argmax(Q[component][:,state]))+np.argmax(Q[component][:,state])
        #print(action,state)
        if action*(Ind_components[component][Ind_components.index[t+1]]-Ind_components[component][Ind_components.index[t]])>0:
            signal_convergence.append(1)
        else :
            signal_convergence.append(0)
        
    return np.mean(signal_convergence)


def Q_train(error,train,epsilon,sub_division):
    global Ind_components
    global Q
    global k_it
    global nu
    
    for iteration in range(num_iterations):  
        print ('{}'.format(iteration+1) +'out of {}'.format(num_iterations))
        for component in  Ind_components.columns: 
            
            for t in range(train):
                return_component=Ind_components[component][Ind_components.index[t]]
                state=LocalizeState(return_component,component,sub_division)
                if random.uniform(0,1)<epsilon[iteration]:
                    
                    action=(1-np.argmax(Q[component][:,state]))+np.argmax(Q[component][:,state])
        
                else:
                    rand=random.randint(0,1)
                    action=(1-rand)+rand
                    
                Q=Q_update(state,action, component,t,nu[iteration,t])
                k_it[component]=action
    return Q


def test_signal(start_date, end_date):
    
    global train_size
    global Ind_components
    global subdivision
    
    signal_convergence={component:[] for component in Ind_components.columns}
    for t in range(start_date, end_date):
        for component in Ind_components.columns:
            return_component=Ind_components[component][Ind_components.index[t]]
            state=LocalizeState(return_component,component,subdivision)
            action=-1*(1-np.argmax(Q[component][:,state]))+np.argmax(Q[component][:,state])
            if action*(Ind_components[component][Ind_components.index[t+1]])>0:
                signal_convergence[component].append(1)
            else :
                signal_convergence[component].append(0)
            
    return signal_convergence

def save_obj(obj, name):
    WD = os.getcwd()
    with open(WD +'/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    WD = os.getcwd()
    with open(WD + '/'+ name + '.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        
        return p

if __name__ == "__main__":
    os.chdir('C:\\Users\\Clement Natta\\Desktop\\PFE')
    return_df=pd.read_csv('RawData.csv',index_col=0)[['Open Price']].pct_change()
    return_df=return_df.drop(return_df.index[0])
    Ind_components=return_df
   # Ind_components=pd.read_csv('Index_returns.csv',index_col=0,header=None)
    #demixing=pd.read_csv('demixing_index.csv',index_col=0)
    #demixing=np.eye(Ind_components.shape[1])    
    
    '''
    
    Testing the Fast ICA 
    
    '''
    
    
    #corr_factors=np.corrcoef(Ind_components.transpose())
    #corr_returns=np.corrcoef(return_df.transpose())
    #cov_factors=np.cov(Ind_components.transpose())
    #mean_factors=np.mean(Ind_components)
    
    #test_statistic= statsmodels.stats.diagnostic.acorr_ljungbox(Ind_components['0'])
    
    
    
    '''
    
      Parameters 
     
    '''
    
    num_bins=100
    num_components=np.shape(return_df)[1]
    num_actions=2
    gamma=1-0.01/365
    bins = np.array(np.arange(1,100,50))
    num_iterations=100
    train_size=1499
    #int(Ind_components.shape[0]*2/3)
    delta=0.005 # given in the paper 
    granularity=5
    epsilon=[0.90/(0.05*k+1) for k in range(num_iterations)]
    epsilon=[0.90 for k in range(num_iterations)]
    
    nu=np.array([[0.9*1/((1+0.01*k)*(i*0.05+1)) for k in  range(train_size)]for i in range(num_iterations)])
    lambda_=0.2
    error=0.0001
    N_u=100
    N_l=100
    
    #Model Initilization
    subdivision={component:0 for component in Ind_components.columns}
    
    for component in subdivision.keys():
            subdivision[component]=initLocalizeState(component)
    
    #k_it={component:-1*(1-init)+1*init for (init,component) in zip(list(bernoulli.rvs(0.5,size=Ind_components.shape[1])),Ind_components.columns)}
    k_it={component:-1*(1-int(init+0.5))+1*int(init+0.5) for (init,component) in zip([np.random.uniform(0,1) for size in Ind_components.columns],Ind_components.columns)}
    
    #Q= {component: np.random.randn(num_actions,2*len(subdivision[component])) for component in Ind_components.columns}
    Q= {component: 0.01*np.random.randn(num_actions,2*len(subdivision[component])) for component in Ind_components.columns}
    eligibility= {component: 0.01*np.zeros((num_actions,2*len(subdivision[component]))) for component in Ind_components.columns}
    
    rewards={component:[] for component in Ind_components.columns}
    portf_return=[]
    
    '''
       Convergence testing at the component level 
    '''
    
        
    
    Q=Q_train_error(num_iterations,train_size,epsilon,subdivision,error)
    
    test_convergence=test_signal(0,train_size)
    save_obj(Q,'Q_error_0.0001_subset_lambda')
    convergence=test_signal(0,train_size)
    np.mean(convergence['Open Price'])
    test=[]

    for lambda_s in range(0,20,1):
        print(lambda_s)
        lambda_=lambda_s/20
        test.append(Q_train_component_error(num_iterations,train_size,epsilon,subdivision,error,'Open Price')[1])
    lambda_s=[x/20 for x in range(0,20,1)]
    plt.xlim(0,1)
    plt.plot(lambda_s,test)
    '''
    average_convergence=[]
     
    for component in convergence.keys():
        average_convergence.append(sum(convergence[component])/len(convergence[component]))
    
    save_obj(average_convergence,'average_con vergence_0.0001_subset_lambda')
    plt.hist(average_convergence) 
    np.mean(average_convergence)
    plt.plot(average_convergence)
    
    prediction_conv_error_lambda=test_signal(train_size,len(Ind_components['0'])-1)
    prediction_avg_conv_error_lambda=[]
    for component in convergence.keys():
        prediction_avg_conv_error_lambda.append(sum(prediction_conv_error_lambda[component])/len(prediction_conv_error_lambda[component]))
    
    plt.plot(prediction_avg_conv_error_lambda)
    plt.hist(prediction_avg_conv_error_lambda)
    np.mean(prediction_avg_conv_error_lambda)

    
    demixing=opt.matrix(np.array(demixing).T.tolist())
    Q=load_obj('Q_error_0.0001_regular_100')
    average_convergence_lambda=load_obj('average_convergence_error_0.001_regular')
    average_convergence=load_obj('average_convergence_error_0.001_lambda')
    
    plt.hist(average_convergence)    
    plt.plot(average_convergence)    
    plt.plot(average_convergence_lambda)    
    
    convergence=test_signal(0,train_size)
    average_convergence=[]
    for component in convergence.keys():
        average_convergence.append(sum(convergence[component])/len(convergence[component]))
    
    prediction_conv_error_100=test_signal(train_size,len(Ind_components['0'])-1)
    prediction_avg_conv_error=[]    
    for component in convergence.keys():
        prediction_avg_conv_error.append(sum(prediction_conv_error_100[component])/len(prediction_conv_error_100[component]))
        
    
    asset_portf_return=[]
    factor_portf_return=[]
    asset_equally_weighted=[]
    factor_equally_weighted=[]
    asset_weights=[]
    factor_weights=[]
                
    lower_bounds={component:[] for component in Ind_components.columns}
    upper_bounds={component:[] for component in Ind_components.columns}
    decision={component:0 for component in Ind_components.columns}
    '''
    rewards_lambda={component:[] for component in Ind_components.columns}
    for t in range(train_size,len(return_df)-1):
        print('{}'.format(t) +' out of {}'.format(len(return_df)))
        
        #if (t-train_size)% 50 ==0:
                
        for component in Ind_components.columns: 
            return_component=Ind_components[component][Ind_components.index[t-1]]
            state=LocalizeState(return_component,component,subdivision)
            action=-1*(1-np.argmax(Q[component][:,state]))+np.argmax(Q[component][:,state])
            rewards[component].append(log(1+((action+1)/2-(1-action)*1/2)*Ind_components[component][Ind_components.index[t]]))
            #portfolio=optimal_portfolio(Ind_components.ix[:t,:],decision,demixing)      
        
        #weights=portfolio['x']
        #print(sum(weights))

    rewards_df_lambda=pd.DataFrame(rewards_lambda).cumsum()

    plt.plot(rewards_lambda['Open Price'])
    plt.plot(rewards_df_lambda['Open Price'])
    
        '''
        
        for component in Ind_components.columns:
            lower_bounds[component].append((1/2*(-1+tanh(decision[component]*N_l))))
            upper_bounds[component].append((1/2*(1+tanh(decision[component]*N_u))))        
            
        
            
        asset_weights.append(np.dot(weights.T,demixing))
        factor_weights.append(list(weights))
        #print(np.shape(weights))
        factor_actual_returns=opt.matrix(Ind_components.ix[t,:])
        asset_actual_returns=opt.matrix(return_df.ix[t,:])
        print(np.size(asset_weights),t-train_size)
        asset_portf_return.append(np.dot(asset_actual_returns.T,(asset_weights[t-train_size]).T)[0][0])
        factor_portf_return.append(np.dot(factor_actual_returns.T,np.array(weights))[0][0])
        asset_equally_weighted.append(1/(np.shape(Ind_components)[1])*sum(asset_actual_returns))
        factor_equally_weighted.append(1/(np.shape(Ind_components)[1])*sum(factor_actual_returns))
        k_it={component:np.sign(x) for (component,x) in zip(Ind_components.columns,weights)}
    '''     
    '''
    

plt.plot([factor_weights[i][0] for i in range(len(factor_weights))])
plt.plot([upper_bounds['0'][i] for i in range(len(upper_bounds['0']))])
plt.plot([lower_bounds['0'][i] for i in range(len(lower_bounds['0']))])
np.argmax(average_convergence)
np.argmax(prediction_avg_conv_error)

mean_factor_weights=[]
for i in range(len(factor_weights[0])):
    mean_factor_weights.append(np.mean([factor_weights[j][i] for j in range(len(factor_weights))]))

abs_mean_factor_weights=[]
for i in range(len(factor_weights[0])):
    abs_mean_factor_weights.append(abs(np.mean([factor_weights[j][i] for j in range(len(factor_weights))])))
   
sum([0 if (upper_bounds['10'][i]>factor_weights[i][10]) else 1 for i in range(len(upper_bounds['10']))  ])
sum([0 if (lower_bounds['10'][i]<factor_weights[i][10]) else 1 for i in range(len(upper_bounds['10']))  ])
        
cumsum_eq_weight=pd.DataFrame(asset_equally_weighted).cumsum()
cumsum_portf=pd.DataFrame(asset_portf_return).cumsum()
pd.DataFrame(asset_portf_return).hist()
plt.plot(asset_equally_weighted)
plt.plot(asset_portf_return)
plt.show()
plt.plot(factor_weights)

# Cumsums plots

plt.plot(cumsum_eq_weight[100:200])
plt.plot(cumsum_portf[100:200])
plt.show()


asset_weights_df=pd.DataFrame(asset_weights)


