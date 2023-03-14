import numpy as np
import pandas as pd

### This part generates sequential data within the fixed-length window ###
'''
Features used in Mauli's model
1) Side chain length: 0, 1, 2, 3, 4, 5, 6 or 7 
where 0 is No Residue, 1 is Glycine, 2 Very Small, 3 Small, 4 Normal, 5 Long, 6 Cycle and 7 Proline from positions −1 to +5
2) Non-polar aliphatic amino acids from positions −3 to −1: 0, 1, 2 or 3
3) Polar positively charged residues from positions −7 to −5: 0, 1, 2 or 3
4) Number of serines and threonines in the -/+10 residue window
5) Flexibility: continuous value from 0 to 1 where 0 is flexible and 1 rigid
6) Secondary structure: 0, 1 or 2 where 0 is not structured, 1 is alpha helix and 2 is beta strand
7) Presence of a proline in +1: 0 or 1 (no or yes)
8) Secondary structure according to phi and psi angles (0 other, 1 beta or 2 alpha)
9) Nature of the site: 0 or 1 where 0 is serine and 1 threonine
'''

def pass_list():
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

amino_acid = {"A":1, "R":2, "N":3, "D":4, "C":5, 
              "E":6, "Q":7, "G":8, "H":9, "I":10, 
              "L":11, "K":12, "M":13, "F":14, "P":15, 
              "S":16, "T":17, "W":18, "Y":19, "V":20}
'''
1)  Alanine (Ala, A)
2)  Arginine (Arg, R)
3)  Asparagine (Asn, N)
4)  Aspartic acid (Asp, D)
5)  Cysteine (Cys, C)
6)  Glutamic acid (Glu, E)
7)  Glutamine (Gln, Q)
8)  Glycine (Gly, G)
9)  Histidine (His, H)
10) Isoleucine (Ile, I)
11) Leucine (Leu, L)
12) Lysine (Lys, K)
13) Methionine (Met, M)
14) Phenylalanine (Phe, F)
15) Proline (Pro, P)
16) Serine (Ser, S)
17) Threonine (Thr, T)
18) Tryptophan (Trp, W)
19) Tyrosine (Tyr, Y)
20) Valine (Val, V)
'''

# def letter_to_token(letter):
#     if letter in dictionary.keys():
#         return dictionary[letter]
#     else:
#         return 0

def make_window(protein_ss, index, start=-10, end=10, marking=False):
    start_index = min(max(index+start, 0), len(protein_ss))
    end_index   = max(min(index+end+1, len(protein_ss)), 0)
#     if marking:
#         sequence = protein_ss['SEQ'].iloc[window_start:window_end].copy()
#         sequence.iloc[index-window_start] = f'"{sequence.iloc[index-window_start]}"'
#         sequence = sequence.sum()        
#     else:
#         sequence = protein_ss['SEQ'].iloc[window_start:window_end].sum()

    window = protein_ss['SEQ'].iloc[start_index:end_index].sum()
    return window

def side_chain(letter):
    if (letter == "G"): #1
        return 'gly' 
    elif (letter == "V" or letter == "A"): #2 Val, Ala
        return 'very_small'
    elif (letter == "S" or letter == "I" or letter == "L" or letter == "T" or letter == "C"): #3 Ser, Thr, Ile, Leu, Cys
        return 'small'
    elif (letter == "D" or letter == "E" or letter == "N" or letter == "Q" or letter == "M"): #4 Asp, Asn, Glu, Gln, Met
        return 'normal'
    elif (letter == "R" or letter == "K"): #5 Arg, Lys
        return 'long'
    elif (letter == "F" or letter == "W" or letter == "Y" or letter == "H"): #6 Phe, Trp, Tyr, His
        return 'cycle'
    elif (letter == "P"): #7
        return 'pro'
    else:
        return 'None' #0
    
def nonpolar_aliphatic(protein_ss, index, start=-3, end=-1): # Non-polar aliphatic AA from -3 ro -1(Ala, Val, Leu, Ile, Pro)
    window = make_window(protein_ss, index, start, end)
    if window:
        nA = window.count("A")
        nV = window.count("V")
        nL = window.count("L")
        nI = window.count("I")
        nP = window.count("P")
        return nA + nV + nL + nI + nP
    else:
        return 0
    

def positively_charged(protein_ss, index, start=-7, end=-5): # count the number of positively charged AA from -7 to -5 (Ard, Lys, His)
    window = make_window(protein_ss, index, start, end)
    if window:
        nR = window.count("R")
        nK = window.count("K")
        nH = window.count("H")
        return nR + nK + nH
    else:
        return 0
    
def S_and_T(protein_ss, index, start=-10, end=10): # Number of serines and threonines in the -/+10 residue window
    window = make_window(protein_ss, index, start, end)
    if window:
        nS = window.count("S")
        nT = window.count("T")
        return nS + nT

    else:
        return 0
    
def is_proline_after(protein_ss, index, after=1): # check whether there is a proline after the site
    if index+after >=0 and index+after <= len(protein_ss)-1:
        return int(protein_ss['SEQ'].iloc[index+after] == 'P')
    else:
        return int(False)

def phi_psi(protein_ss, index):
    phi, psi = protein_ss[['Phi','Psi']].iloc[index]
    if phi > -160 and phi < -50:
        if psi > 100 and psi < 180:
            return "alpha"
        elif psi > -60 and psi < 20:
            return "beta"
        else:
            return "other"
    else:
        return "other"
    
def data_to_sequence(data_x, data_y, window_size):
    ST_idx = np.where((data_x['SEQ_S']==1)|(data_x['SEQ_T']==1))[0]
    input_list = []
    output_list = []
    for idx in ST_idx:
        start_idx = idx - window_size
        end_idx   = idx + window_size + 1
        
        if start_idx < 0:
            zeros = np.zeros((-start_idx,data_x.shape[1]))
            temp  = data_x.iloc[0:end_idx].values
            temp  = np.concatenate([zeros, temp], axis=0)
            
        elif end_idx > len(data_x):
            zeros = np.zeros((end_idx-len(data_x),data_x.shape[1]))
            temp  = data_x.iloc[start_idx:len(data_x)].values
            temp  = np.concatenate([temp, zeros], axis=0)
            
        else:
            temp  = data_x.iloc[start_idx:end_idx].values
            
        input_list.append(temp)
        output_list.append(data_y.iloc[idx].values)
        
    return np.array(input_list), np.array(output_list)