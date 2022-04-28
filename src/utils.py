import pandas as pd
import torch
import numpy as np
import ast
from sklearn.model_selection import train_test_split


def read_from_csv(fpath):
    return(pd.read_csv(fpath, encoding = 'latin1'))

def read_from_excel(fpath):
    return(pd.read_excel(fpath))

def get_train_val_split(df, val_fraction):
    x_train, x_val = train_test_split(df, test_size=val_fraction, shuffle=True, random_state=1000, stratify=df['label'])
    return(x_train, x_val)

def get_param_optimizer(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_parameters

def getoptimizer(model, optimizer, lr):
    if (optimizer == 'adam'):
        opt = torch.optim.Adam(params=model.parameters(), lr=lr)
        # opt = torch.optim.Adam(params=get_param_optimizer(model), lr=lr)
        
    elif (optimizer == 'sgd'):
        opt = torch.optim.SGD(params=model.parameters(), lr=lr)
        # opt = torch.optim.SGD(params=get_param_optimizer(model), lr=lr)

    else:
        raise('Do not support other optimizers currently')

    return(opt) 


def get_prediction_classes(probs, threshold):
    sorted_index_list = []
    sorted_prob_list = []
    for x in probs:
        idx, = np.where(x>=threshold)
        sorted_index = idx[np.argsort(x[idx])][::-1]
        sorted_probs = x[sorted_index]
        sorted_index_list.append(list(sorted_index))
        sorted_prob_list.append(list(sorted_probs))
    return(sorted_index_list, sorted_prob_list)

def extend_multilabels(df):
    new_df = []
    for index, row in df.iterrows():
        if(len(ast.literal_eval(row['label'])) == 1):
            new_df.append((row['text'], ast.literal_eval(row['label'])[0]))
        else:
            new_df.extend((zip([row['text']]*len(ast.literal_eval(row['label'])), ast.literal_eval(row['label']))))
    new_df = pd.DataFrame(new_df, columns = ['text', 'label'])
    return(new_df)    



    