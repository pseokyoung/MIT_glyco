####################################################################################################
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def plot_confusion(y_real, y_pred, title, label=["True", "False"], size=4):
    if y_real.max() == 1:
        cm = confusion_matrix(y_real, y_pred)
        cm = cm.flatten()[::-1].reshape(2,2)
        plt.figure(figsize=(size*1.2,size))
        heatmap(cm, annot=True, fmt='.0f',
                cmap='Blues', 
                xticklabels=label, yticklabels=label)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Real")
        plt.show()
        
        accuracy  = 100*np.array(accuracy_score(y_real, y_pred)).round(4)
        precision = 100*np.array(precision_score(y_real, y_pred, average=None)).round(4)
        recall    = 100*np.array(recall_score(y_real, y_pred, average=None)).round(4)
        f1        = 100*np.array(f1_score(y_real, y_pred, average=None)).round(4)
        print(f"accuracy  : {accuracy}")
        print(f"precision : {precision}")
        print(f"recall    : {recall}")
        print(f"f1_score  : {f1}")