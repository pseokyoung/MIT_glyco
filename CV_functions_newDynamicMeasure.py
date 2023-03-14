# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:43:53 2021

@author: chemegrad2018
"""


import numpy as np
import math
from scipy.spatial import distance
from fastdtw import fastdtw
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
# from sklearn.ensemble.SVM import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#from fastdtw import fastdtw
#from dtaidistance import dtw
from dtw import *
import time





def CVpartition(X, y, Type = 'Re_KFold', K = 5, Nr = 10, random_state = 0):
    '''This function create partition for data for cross validation and bootstrap
    https://scikit-learn.org/stable/modules/cross_validation.html
    
    Input:
    X: independent variables of size N x m np_array
    y: dependent variable of size N x 1 np_array
    type: 'KFold', 'Re_KFold', 'MC' for cross validation
          'TS' for time series cross validation
          
    K: float, 1/K portion of data will be used as validation set, default 10
    Output:partitioned data set
    Nr: Number of repetitions, ignored whtn CV_type = 'KFold', for Re_KFold, it will be Nr * K total
    group: group index for grouped CV
    
    Output:generator (X_train, y_train, X_val, y_val)
    '''
    
    if Type == 'Re_KFold':
        CV = RepeatedStratifiedKFold(n_splits= int(K), n_repeats= Nr, random_state =random_state)
        for train_index, val_index in CV.split(X,y):
            yield (X[train_index], y[train_index], X[val_index], y[val_index])
            
    # elif Type == 'Timeseries':
    #     print('Timeseries')   
    #     TS = TimeSeriesSplit(n_splits=int(K))
    #     for train_index, val_index in TS.split(X):
    #         yield (X[train_index], y[train_index], X[val_index], y[val_index])           
  
    else:
        print('Wrong type specified for data partition')

def dynamic_features(X,X_test):
    #Creates dynamic features based on dynamic time warping distances
    XFeature = np.zeros((len(X),len(X)))
    XtestFeature = np.zeros((len(X_test),len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            XFeature[j,i] = dtw(X[j,:],X[i,:], distance_only = True)
        for k in range(len(X_test)):
            XtestFeature[k,i] = dtw(X_test[k,:],X[i,:], distance_only = True)
    XOut = XFeature
    X_testOut = XtestFeature
    return (XOut, X_testOut)
            
def CV_acc(model_name, X, y, X_test, y_test, cv_type = 'Re_KFold', K_fold = 5, Nr = 10):
    '''This function determines the best hyper_parameter using mse based on CV
    Input:
    model_name: str, indicating which model to use
    X: independent variables of size N x m np_array
    y: dependent variable of size N x 1 np_array
    cv_type: cross_validation type
    K: fold for CV
    Nr: repetition for CV
    **kwargs: hyper-parameters for model fitting, if None, using default range or settings
    
    
    Output: 
    hyper_params: dictionary, contains optimal model parameters based on cross validation
    model: final fited model on all training data
    model_params: np_array m x 1
    mse_train
    mse_test
    yhat_train
    yhat_test
    '''
    if model_name == 'FDA':
        #Cross validation procedure
        solver = ['svd', 'lsqr', 'eigen']
        acc_result = np.zeros((len(solver),1))
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr):
            counter += 1
            for j in range(len(solver)):
                model = LinearDiscriminantAnalysis(solver = solver[j])
                model.fit(X_train,y_train)
                yhat_val = model.predict(X_val)
                acc_val = 1 - np.sum(np.abs(y_val-yhat_val))/np.shape(y_val)[0]
                acc_result[j] += acc_val
        acc_result = acc_result/counter      
        idx = np.argmax(acc_result)
        solver_best = solver[idx]
        hyper_params = {}
        hyper_params['solver'] = solver[idx]
        
        #Run final model with determined hyperparameter(s)
        model = LinearDiscriminantAnalysis(solver = hyper_params['solver'])
        model.fit(X,y)
        yhat_train = model.predict(X)
        yhat_test = model.predict(X_test)
        yprob_train = model.predict_proba(X)
        yprob_test = model.predict_proba(X_test)
        acc_train = 1 - np.sum(np.abs(y-yhat_train))/np.shape(y)[0]
        acc_test = 1 - np.sum(np.abs(y_test-yhat_test))/np.shape(y_test)[0]
        acc_val_best = acc_result[idx]
        return (model, hyper_params, acc_train, acc_test, acc_val_best, yhat_train, yhat_test, yprob_train, yprob_test)

    
    elif model_name == 'QDA':
        #Cross validation procedure
        reg = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 1]
        acc_result = np.zeros((len(reg),1))
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr):
            counter += 1
            for j in range(len(reg)):
                model = QuadraticDiscriminantAnalysis(reg_param = reg[j])
                model.fit(X_train,y_train)
                yhat_val = model.predict(X_val)
                acc_val = 1 - np.sum(np.abs(y_val-yhat_val))/np.shape(y_val)[0]
                acc_result[j] += acc_val
        acc_result = acc_result/counter      
        idx = np.argmax(acc_result)
        reg_best = reg[idx]
        hyper_params = {}
        hyper_params['reg'] = reg_best
        
        #Run final model with determined hyperparameter(s)
        model = QuadraticDiscriminantAnalysis(reg_param = hyper_params['reg'])
        model.fit(X,y)
        yhat_train = model.predict(X)
        yhat_test = model.predict(X_test)
        yprob_train = model.predict_proba(X)
        yprob_test = model.predict_proba(X_test)
        acc_train = 1 - np.sum(np.abs(y-yhat_train))/np.shape(y)[0]
        acc_test = 1 - np.sum(np.abs(y_test-yhat_test))/np.shape(y_test)[0]
        acc_val_best = acc_result[idx]
        return (model, hyper_params, acc_train, acc_test, acc_val_best, yhat_train, yhat_test, yprob_train, yprob_test)
    
    elif model_name == 'SVM':
       #Cross validation procedure
        kernels = ['poly', 'rbf', 'sigmoid']
        Cs = [0.001, 0.01, 0.1, 1, 10 ,50, 100, 500]
        gb = 1/X.shape[1]
        gammas = [gb/50, gb/10, gb/5, gb/2, gb, gb*2, gb*5, gb*10, gb*50]
        acc_result = np.zeros((len(kernels),len(Cs),len(gammas)))
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr):
            counter += 1
            for i in range (len(kernels)):
                for j in range(len(Cs)):
                    for k in range(len(gammas)):
                        model = svm.SVC(C = Cs[j], kernel = kernels[i], gamma = gammas[k])
                        model.fit(X_train,y_train)
                        yhat_val = model.predict(X_val)
                        acc_val = 1 - np.sum(np.abs(y_val-yhat_val))/np.shape(y_val)[0]
                        acc_result[i,j,k] += acc_val
        acc_result = acc_result/counter      
        idx = np.unravel_index(np.argmax(acc_result, axis=None), acc_result.shape)
        kernel = kernels[idx[0]]
        C = Cs[idx[1]]
        gamma = gammas[idx[2]]

        hyper_params = {}
        hyper_params['kernel'] = kernel
        hyper_params['C'] = C
        hyper_params['gamma'] = gamma
        
        #Run final model with determined hyperparameter(s)
        model = svm.SVC(C = hyper_params['C'], kernel = hyper_params['kernel'], gamma = hyper_params['gamma'])
        model.fit(X,y)
        yhat_train = model.predict(X)
        yhat_test = model.predict(X_test)
        acc_train = 1 - np.sum(np.abs(y-yhat_train))/np.shape(y)[0]
        acc_test = 1 - np.sum(np.abs(y_test-yhat_test))/np.shape(y_test)[0]
        acc_val_best = acc_result[idx[0],idx[1],idx[2]]
        return (model, hyper_params, acc_train, acc_test, acc_val_best, yhat_train, yhat_test)
    
    elif model_name == 'LSVM':
       #Cross validation procedure
        Cs = [0.001, 0.01, 0.1, 1, 10 ,50, 100, 500]
        gb = 1/X.shape[1]
        gammas = [gb/50, gb/10, gb/5, gb/2, gb, gb*2, gb*5, gb*10, gb*50]
        acc_result = np.zeros((len(Cs),len(gammas)))
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr):
            counter += 1
            for i in range(len(Cs)):
                for j in range(len(gammas)):
                    model = svm.SVC(C = Cs[i], kernel = 'linear', gamma = gammas[j])
                    model.fit(X_train,y_train)
                    yhat_val = model.predict(X_val)
                    acc_val = 1 - np.sum(np.abs(y_val-yhat_val))/np.shape(y_val)[0]
                    acc_result[i,j] += acc_val
        acc_result = acc_result/counter      
        idx = np.unravel_index(np.argmax(acc_result, axis=None), acc_result.shape)
        C = Cs[idx[0]]
        gamma = gammas[idx[1]]

        hyper_params = {}
        hyper_params['C'] = C
        hyper_params['gamma'] = gamma
        
        #Run final model with determined hyperparameter(s)
        model = svm.SVC(C = hyper_params['C'], kernel = 'linear', gamma = hyper_params['gamma'])
        model.fit(X,y)
        yhat_train = model.predict(X)
        yhat_test = model.predict(X_test)
        acc_train = 1 - np.sum(np.abs(y-yhat_train))/np.shape(y)[0]
        acc_test = 1 - np.sum(np.abs(y_test-yhat_test))/np.shape(y_test)[0]
        acc_val_best = acc_result[idx[0],idx[1]]
        return (model, hyper_params, acc_train, acc_test, acc_val_best, yhat_train, yhat_test)
    
    elif model_name == 'RF':
        #Cross validation procedure
        nEstimators = [10, 50, 100, 200]
        maxDepth = [2,3,5,10,15,20,40]
        minSamplesLeaf = [1,2,3]
        acc_result = np.zeros((len(nEstimators),len(maxDepth),len(minSamplesLeaf)))
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr):
            counter += 1
            for i in range(len(nEstimators)):
                for j in range(len(maxDepth)):
                    for k in range(len(minSamplesLeaf)):
                        model = RandomForestClassifier(n_estimators = nEstimators[i], max_depth = maxDepth[j], min_samples_leaf = minSamplesLeaf[k])
                        model.fit(X_train,y_train)
                        yhat_val = model.predict(X_val)
                        acc_val = 1 - np.sum(np.abs(y_val-yhat_val))/np.shape(y_val)[0]
                        acc_result[i,j,k] += acc_val
        acc_result = acc_result/counter      
        idx = np.unravel_index(np.argmax(acc_result, axis=None), acc_result.shape)
        n_estimator = nEstimators[idx[0]]
        max_depth = maxDepth[idx[1]]
        min_samples_leaf = minSamplesLeaf[idx[2]]

        hyper_params = {}
        hyper_params['n_estimator'] = n_estimator
        hyper_params['max_depth'] = max_depth
        hyper_params['min_samples_leaf'] = min_samples_leaf
        
        #Run final model with determined hyperparameter(s)
        model = RandomForestClassifier(n_estimators = hyper_params['n_estimator'], max_depth = hyper_params['max_depth'], min_samples_leaf = hyper_params['min_samples_leaf'])
        model.fit(X,y)
        yhat_train = model.predict(X)
        yhat_test = model.predict(X_test)
        yprob_train = model.predict_proba(X)
        yprob_test = model.predict_proba(X_test)
        acc_train = 1 - np.sum(np.abs(y-yhat_train))/np.shape(y)[0]
        acc_test = 1 - np.sum(np.abs(y_test-yhat_test))/np.shape(y_test)[0]
        acc_val_best = acc_result[idx[0],idx[1],idx[2]]
        return (model, hyper_params, acc_train, acc_test, acc_val_best, yhat_train, yhat_test, yprob_train, yprob_test)
 
    elif model_name == 'KNN':
       #Cross validation procedure
        nBase = math.sqrt(len(X))
        if nBase *10 <= len(X):
            nNeighbors = [1, round(nBase/10), round(nBase/5), round(nBase/3), round(nBase/2), round(nBase*2), round(nBase*3), round(nBase*5), round(nBase*10)]
        else:
            nNeighbors = [1,2,3,5,8,10,20]
        Weights = ['uniform','distance']
        acc_result = np.zeros((len(nNeighbors),len(Weights)))
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr):
            counter += 1
            for i in range(len(nNeighbors)):
                for j in range(len(Weights)):
                    model = KNeighborsClassifier(n_neighbors = nNeighbors[i], weights = Weights[j])
                    model.fit(X_train,y_train)
                    yhat_val = model.predict(X_val)
                    acc_val = 1 - np.sum(np.abs(y_val-yhat_val))/np.shape(y_val)[0]
                    acc_result[i,j] += acc_val
        acc_result = acc_result/counter      
        idx = np.unravel_index(np.argmax(acc_result, axis=None), acc_result.shape)
        n_neighbors = nNeighbors[idx[0]]
        weights = Weights[idx[1]]

        hyper_params = {}
        hyper_params['n_neighbors'] =  n_neighbors
        hyper_params['weights'] = weights
        
        #Run final model with determined hyperparameter(s)
        model = KNeighborsClassifier(n_neighbors = hyper_params['n_neighbors'], weights = hyper_params['weights'])
        model.fit(X,y)
        yhat_train = model.predict(X)
        yhat_test = model.predict(X_test)
        yprob_train = model.predict_proba(X)
        yprob_test = model.predict_proba(X_test)
        acc_train = 1 - np.sum(np.abs(y-yhat_train))/np.shape(y)[0]
        acc_test = 1 - np.sum(np.abs(y_test-yhat_test))/np.shape(y_test)[0]
        acc_val_best = acc_result[idx[0],idx[1]]
        return (model, hyper_params, acc_train, acc_test, acc_val_best, yhat_train, yhat_test, yprob_train, yprob_test)
    
    elif model_name == 'NNDTW':
        nBase = math.sqrt(len(X))
        if nBase *10 <= len(X):
            nNeighbors = [1,2]
        else:
            nNeighbors = 1
        #Weights = ['uniform','distance']
        acc_result = np.zeros((len(nNeighbors),1))
        # acc_result = np.zeros((len(nNeighbors),len(Weights)))
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr):
            counter += 1
            #for i in range(len(nNeighbors)):
            for i in range(1):
                # for j in range(len(Weights)):
                    # print([i,j])                   

                model = KNeighborsClassifier(n_neighbors = 1,  metric = dtw) #weights = Weights[j],
                print('Model created')
                model.fit(X_train,y_train)
                print('Model fitted')
                yhat_val = model.predict(X_val)
                print('Prediction done')
                acc_val = 1 - np.sum(np.abs(y_val-yhat_val))/np.shape(y_val)[0]
                acc_result[i] += acc_val
        acc_result = acc_result/counter      
        idx = np.unravel_index(np.argmax(acc_result, axis=None), acc_result.shape)
        n_neighbors = nNeighbors[idx[0]]
        #weights = Weights[idx[1]]

        hyper_params = {}
        hyper_params['n_neighbors'] =  n_neighbors
        #hyper_params['weights'] = weights
        hyper_params['metric'] = DTW
        
        #Run final model with determined hyperparameter(s)
        model = KNeighborsClassifier(n_neighbors = hyper_params['n_neighbors'], weights = hyper_params['weights'])
        model.fit(X,y)
        yhat_train = model.predict(X)
        yhat_test = model.predict(X_test)
        acc_train = 1 - np.sum(np.abs(y-yhat_train))/np.shape(y)[0]
        acc_test = 1 - np.sum(np.abs(y_test-yhat_test))/np.shape(y_test)[0]
        acc_val_best = acc_result[idx[0],idx[1]]
        return (model, hyper_params, acc_train, acc_test, acc_val_best, yhat_train, yhat_test)
    
    elif model_name == 'DTWFDA':
        #Cross validation procedure
        solver = ['svd', 'lsqr']
        acc_result = np.zeros((len(solver),1))
        counter = 0
        start = time.time()
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr):
            #Create validation feature
            X_train, X_val = dynamic_features(X_train,X_val)
            # XtrainFeature = np.zeros((len(X_train),len(X_train)))
            # XvalFeature = np.zeros((len(X_val),len(X_train)))
            # for i in range(len(X_train)):
            #     for j in range(len(X_train)):
            #         XtrainFeature[j,i] = dtw(X_train[j,:],X_train[i,:], distance_only = True)
            #     for k in range(len(X_val)):
            #         XvalFeature[k,i] = dtw(X_val[k,:],X_train[i,:], distance_only = True)
            # X_train = XtrainFeature
            # X_val = XvalFeature
            counter += 1
            end = time.time()
            print(end - start)
            for j in range(len(solver)):
                print(solver[j])
                model = LinearDiscriminantAnalysis(solver = solver[j])
                model.fit(X_train,y_train)
                yhat_val = model.predict(X_val)
                acc_val = 1 - np.sum(np.abs(y_val-yhat_val))/np.shape(y_val)[0]
                acc_result[j] += acc_val
                end = time.time()
                print(end - start)
        acc_result = acc_result/counter      
        idx = np.argmax(acc_result)
        solver_best = solver[idx]
        hyper_params = {}
        hyper_params['solver'] = solver[idx]
        
        #Run final model with determined hyperparameter(s)
        #Create testing feature
        X, X_test = dynamic_features(X,X_test)
        # XtrainFeature = np.zeros((len(X),len(X)))
        # XtestFeature = np.zeros((len(X_test),len(X)))
        # for i in range(len(X)):
        #     for j in range(len(X)):
        #         XtrainFeature[j,i] = dtw(X[j,:],X[i,:], distance_only = True)
        #     for k in range(len(X_test)):
        #         XtestFeature[k,i] = dtw(X_test[k,:],X[i,:], distance_only = True)
        # print('Features created')
        # X = XtrainFeature
        # X_test = XtestFeature
        model = LinearDiscriminantAnalysis(solver = hyper_params['solver'])
        model.fit(X,y)
        yhat_train = model.predict(X)
        yhat_test = model.predict(X_test)
        acc_train = 1 - np.sum(np.abs(y-yhat_train))/np.shape(y)[0]
        acc_test = 1 - np.sum(np.abs(y_test-yhat_test))/np.shape(y_test)[0]
        acc_val_best = acc_result[idx]
        end = time.time()
        print(end - start)
        return (model, hyper_params, acc_train, acc_test, acc_val_best, yhat_train, yhat_test)
    
    elif model_name == 'DTWQDA':
        #Cross validation procedure
        reg = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 1]
        acc_result = np.zeros((len(reg),1))
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr):
            #Create validation feature
            X_train, X_val = dynamic_features(X_train,X_val)
            counter += 1
            for j in range(len(reg)):
                print(reg[j])
                model = QuadraticDiscriminantAnalysis(reg_param = reg[j])
                model.fit(X_train,y_train)
                yhat_val = model.predict(X_val)
                acc_val = 1 - np.sum(np.abs(y_val-yhat_val))/np.shape(y_val)[0]
                acc_result[j] += acc_val
        acc_result = acc_result/counter      
        idx = np.argmax(acc_result)
        solver_best = solver[idx]
        hyper_params = {}
        hyper_params['solver'] = solver[idx]
        
        #Run final model with determined hyperparameter(s)
        #Create testing feature
        X, X_test = dynamic_features(X,X_test)
        model = QuadraticDiscriminantAnalysis(reg_param = hyper_params['reg'])
        model.fit(X,y)
        yhat_train = model.predict(X)
        yhat_test = model.predict(X_test)
        acc_train = 1 - np.sum(np.abs(y-yhat_train))/np.shape(y)[0]
        acc_test = 1 - np.sum(np.abs(y_test-yhat_test))/np.shape(y_test)[0]
        acc_val_best = acc_result[idx]
        return (model, hyper_params, acc_train, acc_test, acc_val_best, yhat_train, yhat_test)
    
    elif model_name == 'NNDTW DTWRF DTWSVM DTWKNN':
        acc_result = np.zeros((4,1))
        
        
        nBase = math.sqrt(len(X))
        if nBase *10 <= len(X):
            nNeighbors = [1,2]
        else:
            nNeighbors = 1
        #Weights = ['uniform','distance']
        acc_result_NNDTW = np.zeros((len(nNeighbors),1))
        # acc_result = np.zeros((len(nNeighbors),len(Weights)))
        counter = 0
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr):
            counter += 1
            for i in range(len(nNeighbors)):
                # for j in range(len(Weights)):
                    # print([i,j])                   

                model = KNeighborsClassifier(n_neighbors = 1,  metric = dtw) #weights = Weights[j],
                print('Model created')
                model.fit(X_train,y_train)
                print('Model fitted')
                yhat_val = model.predict(X_val)
                print('Prediction done')
                acc_val = 1 - np.sum(np.abs(y_val-yhat_val))/np.shape(y_val)[0]
                acc_result_NNDTW[i] += acc_val
        acc_result_NNDTW = acc_result_NNDTW/counter      
        idx = np.unravel_index(np.argmax(acc_result_NNDTW, axis=None), acc_result_NNDTW.shape)
        n_neighbors = nNeighbors[idx[0]]
        #weights = Weights[idx[1]]
        
        #NNDTW hyper
        hyper_params_NNDTW = {}
        hyper_params_NNDTW['n_neighbors'] =  n_neighbors
        #hyper_params['weights'] = weights
        hyper_params_NNDTW['metric'] = DTW
        acc_result[0] = acc_result_NNDTW[idx]
     
      
    
        #Cross validation procedure
        #RF
        nEstimators = [10, 50, 100, 200]
        maxDepth = [2,3,5,10,15,20,40]
        minSamplesLeaf = [1,2,3]
        acc_result_RF = np.zeros((len(nEstimators),len(maxDepth),len(minSamplesLeaf)))
        #SVM
        kernels = ['poly', 'rbf', 'sigmoid']
        Cs = [0.001, 0.01, 0.1, 1, 10 ,50, 100, 500]
        gb = 1/X.shape[1]
        gammas = [gb/50, gb/10, gb/5, gb/2, gb, gb*2, gb*5, gb*10, gb*50]
        acc_result_SVM = np.zeros((len(kernels),len(Cs),len(gammas)))
        #KNN
        nBase = math.sqrt(len(X))
        if nBase *10 <= len(X):
            nNeighbors = [1, round(nBase/10), round(nBase/5), round(nBase/3), round(nBase/2), round(nBase*2), round(nBase*3), round(nBase*5), round(nBase*10)]
        else:
            nNeighbors = [1,2,3,5,8,10,20]
        Weights = ['uniform','distance']
        acc_result_KNN = np.zeros((len(nNeighbors),len(Weights)))
        counter = 0
        start = time.time()
        for X_train, y_train, X_val, y_val in CVpartition(X, y, Type = cv_type, K = K_fold, Nr = Nr):
            #Create validation feature
            X_train, X_val = dynamic_features(X_train,X_val)
            counter += 1
            end = time.time()
            print(end - start)
            #RF
            for i in range(len(nEstimators)):
                for j in range(len(maxDepth)):
                    for k in range(len(minSamplesLeaf)):
                        model = RandomForestClassifier(n_estimators = nEstimators[i], max_depth = maxDepth[j], min_samples_leaf = minSamplesLeaf[k])
                        model.fit(X_train,y_train)
                        yhat_val = model.predict(X_val)
                        acc_val = 1 - np.sum(np.abs(y_val-yhat_val))/np.shape(y_val)[0]
                        acc_result_RF[i,j,k] += acc_val
            #SVM
            for i in range (len(kernels)):
                for j in range(len(Cs)):
                    for k in range(len(gammas)):
                        model = svm.SVC(C = Cs[j], kernel = kernels[i], gamma = gammas[k])
                        model.fit(X_train,y_train)
                        yhat_val = model.predict(X_val)
                        acc_val = 1 - np.sum(np.abs(y_val-yhat_val))/np.shape(y_val)[0]
                        acc_result_SVM[i,j,k] += acc_val
            #KNN
            for i in range(len(nNeighbors)):
                for j in range(len(Weights)):
                    model = KNeighborsClassifier(n_neighbors = nNeighbors[i], weights = Weights[j])
                    model.fit(X_train,y_train)
                    yhat_val = model.predict(X_val)
                    acc_val = 1 - np.sum(np.abs(y_val-yhat_val))/np.shape(y_val)[0]
                    acc_result_KNN[i,j] += acc_val
            end = time.time()
            print(end - start)
       
        #RF hyper
        acc_result_RF = acc_result_RF/counter      
        idx = np.unravel_index(np.argmax(acc_result_RF, axis=None), acc_result_RF.shape)
        n_estimator = nEstimators[idx[0]]
        max_depth = maxDepth[idx[1]]
        min_samples_leaf = minSamplesLeaf[idx[2]]
        acc_result[0] = acc_result_RF[idx]

        hyper_params_RF = {}
        hyper_params_RF['n_estimator'] = n_estimator
        hyper_params_RF['max_depth'] = max_depth
        hyper_params_RF['min_samples_leaf'] = min_samples_leaf
        
        #SVM hyper
        acc_result_SVM = acc_result_SVM/counter      
        idx = np.unravel_index(np.argmax(acc_result_SVM, axis=None), acc_result_SVM.shape)
        kernel = kernels[idx[0]]
        C = Cs[idx[1]]
        gamma = gammas[idx[2]]
        acc_result[2] = acc_result_SVM[idx]

        hyper_params_SVM = {}
        hyper_params_SVM['kernel'] = kernel
        hyper_params_SVM['C'] = C
        hyper_params_SVM['gamma'] = gamma
        
        #KNN hyper
        acc_result_KNN = acc_result_KNN/counter      
        idx = np.unravel_index(np.argmax(acc_result_KNN, axis=None), acc_result_KNN.shape)
        n_neighbors = nNeighbors[idx[0]]
        weights = Weights[idx[1]]
        acc_result[3] = acc_result_KNN[idx]

        hyper_params_KNN = {}
        hyper_params_KNN['n_neighbors'] =  n_neighbors
        hyper_params_KNN['weights'] = weights
        
        idx_model = np.argmax(acc_result)
        #Run final model with determined hyperparameter(s)
        #Create testing feature
        X, X_test = dynamic_features(X,X_test)
        print('Features Created')
        #Final model use this
        if idx_model == 0:
            hyper_params = hyper_params_NNDTW
            model = RandomForestClassifier(n_estimators = hyper_params['n_estimator'], max_depth = hyper_params['max_depth'], min_samples_leaf = hyper_params['min_samples_leaf'])
        elif idx_model == 1:
            hyper_params = hyper_params_RF
            model = RandomForestClassifier(n_estimators = hyper_params['n_estimator'], max_depth = hyper_params['max_depth'], min_samples_leaf = hyper_params['min_samples_leaf'])
        elif idx_model == 2:
            hyper_params = hyper_params_SVM
            model = svm.SVC(C = hyper_params['C'], kernel = hyper_params['kernel'], gamma = hyper_params['gamma'])
        elif idx_model == 3:
            hyper_params = hyper_params_KNN
            model = KNeighborsClassifier(n_neighbors = hyper_params_NNDTW['n_neighbors'], metric = hyper_params_NNDTW['metric'])
            
        model.fit(X,y)
        yhat_train = model.predict(X)
        yhat_test = model.predict(X_test)
        acc_train = 1 - np.sum(np.abs(y-yhat_train))/np.shape(y)[0]
        acc_test = 1 - np.sum(np.abs(y_test-yhat_test))/np.shape(y_test)[0]
        acc_val_best = acc_result[idx_model]
        end = time.time()
        print(end - start)
        return (model, hyper_params, acc_train, acc_test, acc_val_best, yhat_train, yhat_test)