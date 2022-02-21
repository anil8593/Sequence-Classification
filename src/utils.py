import pandas as pd
import torch
from sklearn.model_selection import train_test_split

def read_from_csv(fpath):
    return(pd.read_csv(fpath, encoding = 'latin1'))

def read_from_excel(fpath):
    return(pd.read_excel(fpath))

def get_train_val_split(df, val_fraction):
    x_train, x_val = train_test_split(df, test_size=val_fraction, shuffle=True, random_state=1000, stratify=df['label'])
    return(x_train, x_val)

def getoptimizer(model, optimizer, lr):
    if (optimizer == 'adam'):
        opt = torch.optim.Adam(params=model.parameters(), lr=lr)
    
    elif (optimizer == 'sgd'):
        opt = torch.optim.SGD(params=model.parameters(), lr=lr)

    else:
        raise('Do not support other optimizers currently')

    return(opt) 
    



    