import pandas as pd
import numpy as np

####################################################################################################
x_cts_original = ['ASA', 'Phi', 'Psi', 'Theta(i-1=>i+1)', 'Tau(i-2=>i+2)', 'HSE_alpha_up', 'HSE_alpha_down', 
                        'P(C)', 'P(H)', 'P(E)', 'flexibility']
x_cat_original = ['SEQ', 'SS']

x_cts_window   = ['Proline', 'flexibility']
x_cat_window   = ['SEQ', 'nS/nT', 'nAli', 'nPos', 'phi_psi', 'SS', 
                  'side_-1', 'side_1', 'side_2', 'side_3','side_4', 'side_5']
y_label = ['positivity']

class xy_variables():
    def __init__(self):
        self.x_cts_original = x_cts_original
        self.x_cat_original = x_cat_original
        self.x_cts_window = x_cts_window
        self.x_cat_window = x_cat_window
        self.y_label = y_label

####################################################################################################
def call_pass_list():
    pass_list = ["P24622_2", "Q91YE8_2", # these proteins have positive sites which are out of bound
                 'Q8WWM7']               # this protein does not match between spider and dynamine
    return pass_list

def exclude_list(): # these proteins do not match with the sequence between Mauli and AlphaFold
    exclude_list = [
    'Q62381_2', # 1
    'Q69ZI1_3', # 2
    'Q80TI1_2', # 3
    'Q80TR8_4', # 4
    'Q80YE7_2', # 5
    'Q91YE8_2', # 6
    'Q8BXL9_2'  # 7
    ] 
    return exclude_list
####################################################################################################
def df_to_dummy(data, x_cts=[], x_cat=[], y_label=[]):
    data_x = pd.get_dummies(data[x_cts+x_cat], columns=x_cat)
    data_y = data[y_label]
    
    print(f"dummy x shape: {data_x.shape}")
    print(f"dummy y shape: {data_y.shape}")
    
    return data_x, data_y

onehot_list = {
    'SEQ': ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'],
    'SS': ['C', 'H', 'E'],
    'nS/nT': list(range(1,22)),
    'nAli': list(range(1,4)),
    'nPos': list(range(1,4)),
    'phi_psi': ['other', 'beta', 'alpha'],
    'side_-1': ['gly', 'very_small', 'small', 'normal', 'long', 'cycle', 'pro'],
    'side_1': ['gly', 'very_small', 'small', 'normal', 'long', 'cycle', 'pro'],
    'side_2': ['gly', 'very_small', 'small', 'normal', 'long', 'cycle', 'pro'],
    'side_3': ['gly', 'very_small', 'small', 'normal', 'long', 'cycle', 'pro'],
    'side_4': ['gly', 'very_small', 'small', 'normal', 'long', 'cycle', 'pro'],
    'side_5': ['gly', 'very_small', 'small', 'normal', 'long', 'cycle', 'pro']
}

def to_onehot(value, value_list):
    N = len(value_list)
    onehot = np.eye(N+1, N, -1)
    if value in value_list:
        return onehot[value_list.index(value)+1]
    else:
        return onehot[0]
    
def custom_dummy(data, x_cts=[], x_cat=[], y_label=[]):
    data_x = data[x_cts]
    data_y = data[y_label]
    
    for column in x_cat:
        onehot = np.array( data[column].apply(lambda x: to_onehot(x, onehot_list[column])).to_list() ) 
        data_onehot = pd.DataFrame(onehot, columns = [f"{column}_{x}" for x in onehot_list[column]])
        
        data_x = pd.concat([data_x, data_onehot], axis=1)
    return data_x, data_y

####################################################################################################
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_split(data_x, data_y, n_splits=1, test_size=0.2, random_state=1, scale_x=[], scale_y=[]):
        
    if type(data_x) not in (np.ndarray, pd.core.frame.DataFrame):
        return print("data type must be ndarray or DataFrame")
    
    elif type(data_x) == pd.core.frame.DataFrame:
        data_x = data_x.values
        data_y = data_y.values
        
    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
#     print(f"randoms state for the splitter: {random_state}")
    
    for train_idx, test_idx in split.split(data_x, data_y):
        train_x, train_y = data_x[train_idx], data_y[train_idx]
        test_x,  test_y  = data_x[test_idx],  data_y[test_idx]

    if len(scale_x):
        if train_x.ndim == 2:
            train_sc, test_sc = train_x[:,:len(scale_x)], test_x[:,:len(scale_x)]
            x_min, x_max = train_sc.min(0), train_sc.max(0)
            train_x[:,:len(scale_x)] = (train_sc-x_min)/(x_max-x_min)
            test_x[:,:len(scale_x)]  = (test_sc-x_min) /(x_max-x_min)
            
        elif train_x.ndim == 3:
            train_sc, test_sc = train_x[:,:,:len(scale_x)], test_x[:,:,:len(scale_x)]
            x_min, x_max = train_sc.min(0).min(0), train_sc.max(0).max(0)
            train_x[:,:,:len(scale_x)] = (train_sc-x_min)/(x_max-x_min)
            test_x[:,:,:len(scale_x)]  = (test_sc-x_min) /(x_max-x_min)

    if len(scale_y):
        if train_y.ndim == 2:
            train_sc, test_sc = train_y[:,:len(scale_y)], test_y[:,:len(scale_y)]
            y_min, y_max = train_sc.min(0), train_sc.max(0)
            train_y[:,:len(scale_y)] = (train_sc-y_min)/(y_max-y_min)
            test_y[:,:len(scale_y)]  = (test_sc-y_min) /(y_max-y_min)
            
        elif train_y.ndim == 3:
            train_sc, test_sc = train_y[:,:,:len(scale_x)], test_y[:,:,:len(scale_x)]
            y_min, y_max = train_sc.min(0).min(0), train_sc.max(0).max(0)
            train_y[:,:,:len(scale_y)] = (train_sc-y_min)/(y_max-y_min)
            test_y[:,:,:len(scale_y)]  = (test_sc-y_min) /(y_max-y_min)
            
    print("train/test dataset:", type(data_x))
    print()
    print("train:", train_x.shape, train_y.shape)
    print("check scale:", train_x.min(), train_x.max())
    print()
    print("test:", test_x.shape, test_y.shape)   
    print("check scale:", test_x.min(), test_x.max())
            
    return train_x, train_y, test_x, test_y, train_idx, test_idx

####################################################################################################
import random

def up_sampling(train_x, train_y, random_state=1):
    index_pos = np.where(train_y == 1)[0]
    index_neg = np.where(train_y == 0)[0]

#     print(f"randoms state for generating up-sampling index: {random_state}")
    random.seed(random_state)
    index_up = [random.choice(index_pos) for _ in range(len(index_neg))] # get samples from positive sites as much as the number of negative sites

    upsample_x, sample_x = train_x[index_up], train_x[index_neg]
    upsample_y, sample_y = train_y[index_up], train_y[index_neg]

    concat_x = np.concatenate([upsample_x, sample_x], axis=0)
    concat_y = np.concatenate([upsample_y, sample_y], axis=0)

#     print(f"randoms state for shuffling the up-sampled dataset: {random_state}")
    shuffle_index = np.arange(len(concat_x))
    np.random.seed(random_state)
    np.random.shuffle(shuffle_index)
    concat_x = concat_x[shuffle_index]
    concat_y = concat_y[shuffle_index]
    
    print("up-sampled train dataset:", concat_x.shape, concat_y.shape)
    return concat_x, concat_y
####################################################################################################