# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:06:06 2021

@author: chemegrad2018
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pingouin as pg
import random
import statsmodels.stats.api as sms
import statsmodels.api as sm
from sklearn import neighbors

def dynamic_correlation(x, plot=1, round_number = 0, alpha = 0.01, freq = 1):
    """
    x: time series data of interests of size Nx1
    plot: flag for plotting
    alpha: significance level
    freq: sampling frequency of the time series Hz
    
    Output:
        significant lags for ACF, PACF and CCF
    """
    
    #Dickey-Fuller Tests for statinonary, below alpha is staionary
    # x=x.flatten()
    # xdf = sm.tsa.stattools.adfuller(x,1)
    # if xdf[1]> alpha:
    #     print('x is not stationary')
        
     
    #ACF
    [acf, confint, qstat, acf_pvalues] = sm.tsa.stattools.acf(x, qstat = True, alpha = alpha)
    acf_detection = acf_pvalues < alpha #Ljung-Box Q-Statistic
    acf_lag = [i for i,u in enumerate(acf_detection) if u == True] 

    #PACF
    [pacf, confint_pacf] = sm.tsa.stattools.pacf(x, alpha = alpha)
    pacf_lag = [i for i,u in enumerate(pacf) if abs(u)>2.576/np.sqrt(x.shape[0])]
    
    return (acf_lag, pacf_lag)

def nonlinearity_assess(Xtrain,Ytrain,nNonlin = 50000):
    # Calculate Nonlinearity
    random.seed(0)
    M = np.shape(Xtrain)[1] # the number of features in each sample
    N = np.shape(Xtrain)[0] # the number of samples

    xknum = []
    alpha = []
    for i in range (0,N):
        n = random.randrange(0,len(Xtrain))
        xknum.append(n)
        alphai = random.uniform(0,1)
        alpha.append(alphai)
        
    count = 0
    xlnum = []
    
    while count < N: 
        xldummy = random.randrange(0,len(Xtrain))
        if Ytrain[xldummy] == Ytrain[xknum[count]] and xknum[count] != xldummy:
            count += 1
            xlnum.append(xldummy)
        else:
            continue
            
    xk = Xtrain[xknum,:]
    yk = Ytrain[xknum,:]
    xl = Xtrain[xlnum,:]
    
    xBar = np.zeros((N,M))
    for i in range (0,N):
        xBar[i,:] = alpha[i]*xk[i,:] + (1-alpha[i])*xl[i,:]
    
    KNNclf = neighbors.KNeighborsClassifier(n_neighbors = 1)
    KNNclf.fit(Xtrain, np.ravel(Ytrain))
    Ypredict = KNNclf.predict(xBar)
    NonlinMeasure = sum(abs(np.ravel(yk)-Ypredict))/N
    
    # Calculate overlap measure
    sumOverlap = 0
    
    print('start calculating overlap measure')
    for i in range(0,N):
        if i % 1000 == 0:
            print("count:", i)
        nbrs = neighbors.NearestNeighbors(n_neighbors = 1)
        nbrs.fit(np.concatenate((Xtrain[0:i,:],Xtrain[i+1:-1,:]),axis = 0))
        idx = nbrs.kneighbors(Xtrain[i,:].reshape(1,M),1,return_distance = False)
        if idx >= i: 
            idx += 1   
        if Ytrain[i] == Ytrain[idx]: 
            sumOverlap += 1
    print('end calculating overlap measure')
    OverlapMeasure = 1-sumOverlap/N;
    
    Nonlinearity = NonlinMeasure/OverlapMeasure
    return Nonlinearity




def normality_assess(Xtrain,Ytrain,alpha = 0.05):
    #Calculate normality for each class
    unique = np.unique(Ytrain)
    pvalMax = 0
    
    for i in range(len(unique)):
        idxN, dummy = np.where(Ytrain == unique[i])
        hz, pval, normal = pg.multivariate_normality(Xtrain[idxN,:], alpha = 0.05)
        if pval > pvalMax:
            pvalMax = pval
    
    NormalityMeasure = pvalMax
    return NormalityMeasure

def dynamic_assess(y,yprob_train,nlag = None,alpha = 0.01,round_number = 0):
    if nlag is None:
        if y.shape[0]<22:
            nlag = np.round(y.shape[0]/2)-2
        elif y.shape[0]<40:
            nlag = 10
        elif y.shape[0] >200:
            nlag = 50
        else:
            nlag = y.shape[0]//4
            
    residual = y-yprob_train
    #Calculate dynamic measure (preliminary for all inputs)
    fig = plt.figure(figsize=(5,3))
    ax1 = fig.add_subplot(111)    
    fig = sm.graphics.tsa.plot_acf(residual, lags=nlag, ax=ax1, alpha= alpha, fft=False)
    for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels()):
        item.set_fontsize(14)
    ax1.set_xlabel('Lag')
    plt.tight_layout()
    plt.savefig('ACF_' + str(round_number)+'.png', dpi = 600,bbox_inches='tight')
    
    #partial autocorrelation
    fig = plt.figure(figsize=(5,3))
    ax2 = fig.add_subplot(111)    
    fig = sm.graphics.tsa.plot_pacf(residual, lags=nlag, method='ywmle', ax=ax2, alpha= alpha)
    for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label] + ax2.get_xticklabels() + ax2.get_yticklabels()):
        item.set_fontsize(14)
    ax2.set_xlabel('Lag')
    plt.tight_layout()
    plt.savefig('PACF_' + str(round_number)+'.png', dpi = 600,bbox_inches='tight')
    
    #ACF
    [acf, confint, qstat, acf_pvalues] = sm.tsa.stattools.acf(residual, nlags=nlag,qstat = True, alpha = alpha, fft = False)
    # acf_detection = acf_pvalues < (alpha/nlag) #Ljung-Box Q-Statistic
    # acf_lag = [i for i,x in enumerate(acf_detection) if x == True] 
    acf_lag = [i for i,x in enumerate(acf) if x<confint[i][0] or x>confint[i][1]]

    #PACF
    [pacf, confint_pacf] = sm.tsa.stattools.pacf(residual, nlags=nlag, alpha = alpha)
    pacf_lag = [i for i,x in enumerate(pacf) if x<confint_pacf[i][0] or x>confint_pacf[i][1]]
    

    if acf_lag != [] or pacf_lag != []:
        int_dynamics = 1
    else:
        int_dynamics = 0
    return int_dynamics
    # M = np.shape(Xtrain)[1]
    # N = np.shape(Xtrain)[0]
    # count = 0
    # for i in range(M):
    #     acf_lag, pacf_lag = dynamic_correlation(Xtrain[:,i])
    #     if len(pacf_lag) > 1:
    #         count +=1
    # DynamicMeasure = count/M
    # return DynamicMeasure