import os
import numpy as np
import pandas as pd
#from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
from skimage.filters import threshold_otsu

def es_consecutivo(lista):
    n = len(lista)
    suma = n * min(lista) + n * (n - 1) / 2 if n > 0 else 0
    return sum(lista) == suma

def load_data(directory,x_name,y_name,conf_channel=3, pandas=False, seql=0):
    
    print("Loading x_data... ", end=" ")
    
    x_data = np.load(directory+x_name, allow_pickle=True) # datos
    
    if conf_channel==1: data_x = x_data[:,5]-x_data[:,9]#resta del canal 6 y 10, ofrecen más información sobre ataques epilepticos
    elif conf_channel==2: data_x = x_data.reshape(x_data.shape[0],x_data.shape[1]*x_data.shape[2]) #concatenacion 21 canales 
    elif conf_channel==3: data_x = x_data.mean(axis=1) #media aritmetica canales

    else: print('ERROR LOAD TYPE!!')
    
    print(f'DONE  Shape: {x_data.shape}')
    
    
    
    print("Loading y_data... ", end=" ")
    y_data = np.load(directory+y_name, allow_pickle=True) # metadatos
    data_y = y_data 
    #data_y = y_data [:,7].astype('int')
    print(f'DONE  Shape: {y_data.shape}')
    
    
    print("Tranform data... ", end=" ")                     
    
    if pandas:
        print("Loading pandas data_x... ")
        data_x=pd.DataFrame(data_x)
        print("Loading pandas data_y... ")
        data_y = pd.DataFrame(data_y, columns = ['type','name','filename','pre1','pre2','id_eeg_actual','id_eeg_all','label'])
        data_y.label=data_y.label.astype('int')
        
        #rus = RandomUnderSampler(sampling_strategy=0.5)
        #data_x_r, data_y_r= rus.fit_resample(data_x, data_y)
              
        
    
    #tranformamos [NSamples x 128] a [NSamples x LSeq x 128]
    if seql != 0:
        nSamp =int(data_x.shape[0]/seql)
        data_x = data_x[:nSamp*seql].reshape(-1,seql,128)
        data_y = data_y[:nSamp*seql,-1].reshape(-1,seql)
    
    print(f'DONE  Shape: {data_x.shape} \n')
    
    return data_x, data_y












