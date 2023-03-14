import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

import time
import random

####################################################################################################
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def class_scores(y_real, y_pred, rounding=4, average=None):
    accuracy  = 100*np.array(accuracy_score(y_real, y_pred)).round(rounding)
    precision = 100*np.array(precision_score(y_real, y_pred, average=average)).round(rounding)
    recall    = 100*np.array(recall_score(y_real, y_pred, average=average)).round(rounding)
    f1        = 100*np.array(f1_score(y_real, y_pred, average=average)).round(rounding)
    
    return accuracy, precision, recall, f1
####################################################################################################
class MLP():
    def __init__(self, x_dim, y_dim,
                 model_name = 'MLP'):
        self.x_dim      = x_dim 
        self.y_dim      = y_dim
        self.n_layers   = None
        self.n_neurons  = None
        self.model      = None
        self.model_name = model_name
        self.epoch      = None
        self.time       = None
        self.loss       = None
        self.val_loss   = None
        
    def build(self, n_layers, n_neurons, activation='sigmoid',
              loss = 'binary_crossentropy', metrics= ['accuracy'],
              optimizer='Adam', lr=0.001, print_model=False):
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        
        
        model_input = Input(shape=(self.x_dim,), name='model_input')
        for i in range(n_layers):
            if i==0:
                dense_output = Dense(n_neurons, name=f"dense_{i+1}")(model_input)
                
            else: 
                dense_output = Dense(n_neurons, name=f"dense_{i+1}")(dense_output)
        model_output = Dense(self.y_dim, name=f"model_output", activation=activation)(dense_output)  
        
        model = Model(model_input, model_output)
        if optimizer in ('Adam', 'adam'):
            optimizer = Adam(learning_rate = lr, beta_1=0.9, beta_2=0.999)
        else:
            print("enter valid optimizer")
        model.compile(loss=loss, optimizer = optimizer, metrics=metrics)
        self.model = model
        if print_model:
            print(f"n_layers: {n_layers}")
            print(f"n_neurons: {n_neurons}")
            print(f"Model has been generated: {self.model.summary()}")
        
    def train(self, train_x, train_y, valid_size=None, valid_data=[],verbose=0, save_path=False, seed=1, 
              epochs=10000, early=True, patience=30, monitor='val_loss'):
        callbacks = []
        if early:
            early_stopping_cb = EarlyStopping(patience=patience, restore_best_weights=True, monitor=monitor)
            callbacks.append(early_stopping_cb)
        
        time_start  = time.time()
        random.seed(seed)
        if valid_size:
            history = self.model.fit(train_x, train_y, verbose=verbose,
                                    epochs=epochs, callbacks=callbacks,
                                    validation_split= valid_size)
        elif valid_data:
            history = self.model.fit(train_x, train_y, verbose=verbose,
                                    epochs=epochs, callbacks=callbacks,
                                    validation_data=(valid_data[0], valid_data[1]))
        else:
            print('warning: define validation dataset')
            
        time_end    = time.time()
        time_elapse = round((time_end - time_start)/60, 3)
            
        self.epoch    = np.array(history.history[monitor]).argmin()
        self.time     = time_elapse
        self.loss     = history.history['loss'][self.epoch]
        self.val_loss = history.history['val_loss'][self.epoch]
        
        history       = pd.DataFrame([[self.epoch, self.time, self.loss, self.val_loss]], 
                                     columns=['epoch', 'time', 'loss', 'val_loss'])
        if save_path:
            self.model.save_weights(save_path)
            history.to_csv(save_path[:-2]+'csv', index=False)
            print(f"model has been saved to {save_path}")
        else:
            print("model training finished")        
        
        return history
    
    def load_model(self, load_path):
        self.model.load_weights(load_path)
        print(f"model has been restored from {load_path}")
        
        history = pd.read_csv(load_path[:-2]+'csv', header=0)
        self.epoch    = history['epoch'][0]
        self.time     = history['time'][0]
        self.loss     = history['loss'][0]
        self.val_loss = history['val_loss'][0]
            
    def evaluate(self, eval_x, eval_y, 
                 rounding=4, average=None):
        eval_loss  = self.model.evaluate(eval_x, eval_y, verbose=0)[0]
        prediction = self.model.predict(eval_x, verbose=0).round(0).astype(int)
        accuracy, precision, recall, f1 = class_scores(eval_y, prediction, 
                                                       rounding=rounding, average=average)
        return eval_loss, accuracy, precision, recall, f1
####################################################################################################
class RNN(MLP):
    def __init__(self, history_size, x_dim, y_dim,
                 model_name = 'RNN'):
        super().__init__(x_dim, y_dim,
                         model_name)
        self.rnn_layers   = None
        self.rnn_neurons  = None
        self.dnn_layers   = None
        self.dnn_neurons  = None
        self.history_size = history_size

    def build(self, rnn_layers, rnn_neurons, dnn_layers, dnn_neurons, activation='sigmoid',
              loss = 'binary_crossentropy', metrics= ['accuracy'],
              optimizer='Adam', lr=0.001, print_model=False):

        self.rnn_layers  = rnn_layers
        self.rnn_neurons = rnn_neurons
        self.dnn_layers  = dnn_layers
        self.dnn_neurons = dnn_neurons
        
        model_input = Input(shape=(self.history_size, self.x_dim), name='model_input')

        # encoder module
        if rnn_layers == 1:
            rnn_output, state_h, state_c = LSTM(rnn_neurons, return_state=True, name='rnn_1')(model_input)
            # encoder_states = [state_h, state_c]

        else:
            for i in range(rnn_layers):
                #first encoder layer
                if i==0: 
                    rnn_output = LSTM(rnn_neurons, return_sequences=True, name="encoder_1")(model_input)
                #mediate encoder layer
                elif i < rnn_layers-1: 
                    rnn_output = LSTM(rnn_neurons, return_sequences=True, name=f"encoder_{i+1}")(rnn_output)
                #last encoder layer
                else: 
                    rnn_output, state_h, state_c  = LSTM(rnn_neurons, return_state=True, name=f"encoder_{i+1}")(rnn_output)
                    # encoder_states = [state_h, state_c]

        # dense module
        if dnn_layers == 1:
            dnn_output = Dense(dnn_neurons, name='dense_1')(rnn_output)
        else:
            for i in range(dnn_layers):
                #first dense layer

                if i==0:
                    dnn_output = Dense(dnn_neurons, name='dense_1')(rnn_output)
                #mediate encoder layer
                else:
                    dnn_output = Dense(dnn_neurons, name=f'dense_{i+1}')(dnn_output)
        model_output = Dense(self.y_dim, activation=activation, name=f'model_output')(dnn_output)
        
        # model compile
        model = Model(model_input, model_output)
        if optimizer in ('Adam', 'adam'):
            optimizer = Adam(learning_rate = lr, beta_1=0.9, beta_2=0.999)
        else:
            print("enter valid optimizer")
        model.compile(loss=loss,optimizer = optimizer, metrics=metrics)
        self.model = model
        if print_model:
            print(f"n_layers: {n_layers}")
            print(f"n_neurons: {n_neurons}")
            print(f"Model has been generated: {self.model.summary()}")
    
####################################################################################################
class CNN1D(MLP):
    def __init__(self, history_size, x_dim, y_dim,
                 model_name = 'CNN'):
        super().__init__(x_dim, y_dim,
                         model_name)
        self.cnn_layers   = None
        self.cnn_neurons  = None
        self.dnn_layers   = None
        self.dnn_neurons  = None
        self.history_size = history_size

    def build(self, filter_kernel = [(64, 5), (128, 3)], dnn_neurons=[128, 64], cnn_act = 'relu', activation='sigmoid',
              loss = 'binary_crossentropy', metrics= ['accuracy'],
              optimizer='Adam', lr=0.001, print_model=False):

        self.filter_kernel  = filter_kernel
        self.dnn_neurons = dnn_neurons
        
        model_input = Input(shape=(self.history_size, self.x_dim), name='model_input')

        # encoder module
        if len(filter_kernel) == 1:
            conv = Conv1D(filters = filter_kernel[0][0], kernel_size=filter_kernel[0][1], activation=cnn_act, name='cnn_1')(model_input)
            pool = MaxPooling1D(pool_size=2)(conv)
            

        else:
            for i, (filters, kernel_size) in enumerate(filter_kernel):
                #first encoder layer
                if i==0: 
                    conv = Conv1D(filters = filters, kernel_size=kernel_size, activation=cnn_act, name='cnn_1')(model_input)
                    pool = MaxPooling1D(pool_size=2)(conv)
                #mediate encoder layer
                else: 
                    conv = Conv1D(filters = filters, kernel_size=kernel_size, activation=cnn_act, name=f'cnn_{i+1}_1')(conv)
                    conv = Conv1D(filters = filters, kernel_size=kernel_size, activation=cnn_act, name=f'cnn_{i+1}_2')(conv)
                    pool = MaxPooling1D(pool_size=2)(conv)
        
        flat = Flatten()(pool)
        
        # dense module
        if len(dnn_neurons) == 1:
            dnn_output = Dense(dnn_neurons[0], name='dense_1')(flat)
        else:
            for i, neurons in enumerate(dnn_neurons):
                #first dense layer
                if i==0:
                    dnn_output = Dense(neurons, name='dense_1')(flat)
                #mediate encoder layer
                else:
                    dnn_output = Dense(neurons, name=f'dense_{i+1}')(dnn_output)
                    
        model_output = Dense(self.y_dim, activation=activation, name=f'model_output')(dnn_output)
        
        # model compile
        model = Model(model_input, model_output)
        if optimizer in ('Adam', 'adam'):
            optimizer = Adam(learning_rate = lr, beta_1=0.9, beta_2=0.999)
        else:
            print("enter valid optimizer")
        model.compile(loss=loss,optimizer = optimizer, metrics=metrics)
        self.model = model
        if print_model:
            print(f"n_layers: {n_layers}")
            print(f"n_neurons: {n_neurons}")
            print(f"Model has been generated: {self.model.summary()}")
    
####################################################################################################
class SGL(MLP):
    def __init__(self, x_dim, y_dim,
                 model_name = 'SGL'):
        super().__init__(x_dim, y_dim,
                         model_name)

    def build(self, n_layers, n_neurons, regularizer='l1', activation='sigmoid',
              loss = 'binary_crossentropy', metrics= ['accuracy'],
              optimizer='Adam', lr=0.001, print_model=False):
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        
        model_input = Input(shape=(self.x_dim,), name='model_input')
        for i in range(n_layers):
            if i==0:
                dense_output = Dense(n_neurons, kernel_regularizer=regularizer, name=f"dense_{i+1}")(model_input)
                
            else: 
                dense_output = Dense(n_neurons, kernel_regularizer=regularizer, name=f"dense_{i+1}")(dense_output)
        model_output = Dense(self.y_dim, name=f"model_output", kernel_regularizer=regularizer, activation=activation)(dense_output)  
        
        model = Model(model_input, model_output)
        if optimizer in ('Adam', 'adam'):
            optimizer = Adam(learning_rate = lr, beta_1=0.9, beta_2=0.999)
        else:
            print("enter valid optimizer")
        model.compile(loss=loss, optimizer = optimizer, metrics=metrics)
        self.model = model
        if print_model:
            print(f"n_layers: {n_layers}")
            print(f"n_neurons: {n_neurons}")
            print(f"Model has been generated: {self.model.summary()}")
    
        