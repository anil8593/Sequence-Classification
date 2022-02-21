import pandas as pd
import numpy as np
from transformers.models.auto.configuration_auto import AutoConfig
import utils
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


def start_training(args):
    
    ######## Model run parameters ########
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    require_val_split = args.require_val_split
    val_fraction = args.val_fraction
    model_name = args.basemodel_name
    device = 'cuda' if args.use_gpu else 'cpu'
    optimizer = args.optimiser
    learning_rate = args.lr
    eval_after_train_epoch = args.eval_after_train_epoch

    ########## Reading the data ###########

    df = utils.read_from_csv(os.path.join('input',args.train_filename))

    ########## Data splitting #########

    if (require_val_split):
        if (val_fraction>0):
            x_train, x_val = utils.get_train_val_split(df, val_fraction)
            train_texts, val_texts = x_train['text'].tolist(), x_val['text'].tolist()
            train_targets, val_targets = x_train['label'].tolist(), x_val['label'].tolist()

        else:
            raise ('Please provide valid fraction') 

    else:
        train_texts = df['text'].tolist()
        train_targets = df['label'].tolist()

    num_labels = len(set(train_targets))
    uniq_labels = list(set(train_targets))

    ########## Loading the model and tokenizer ########

    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.to(device)


    ########## Input encoding #########

    encoded_train_texts = tokenizer.batch_encode_plus(batch_text_or_text_pairs=train_texts,
                                                      add_special_tokens=True,
                                                      padding='max_length',
                                                      truncation=True,
                                                      is_split_into_words=False,
                                                      return_tensors='pt')
    
    if (require_val_split):
        encoded_val_texts = tokenizer.batch_encode_plus(batch_text_or_text_pairs=val_texts,
                                                    add_special_tokens=True,
                                                    padding='max_length',
                                                    truncation=True,
                                                    is_split_into_words=False,
                                                    return_tensors='pt')
    
    ########### Target encoding #############

    if (num_labels == 2):
        target_encoder = LabelEncoder()
        target_encoder.fit(train_targets)
        model.config.num_labels = num_labels
    
    elif (num_labels > 2):
        target_encoder = OneHotEncoder()
        target_encoder.fit(np.array(train_targets).reshape(-1,1))
        model.config.num_labels = num_labels

    ############# Converting to torch datasets ##########

    if (num_labels == 2):
        encoded_train_targets = torch.tensor(target_encoder.transform(train_targets), dtype=torch.long)
    
    elif(num_labels > 2):
        encoded_train_targets = torch.tensor(target_encoder.transform(np.array(train_targets).reshape(-1,1)).toarray())

    train_dataset = torch.utils.data.TensorDataset(encoded_train_texts.input_ids, 
                                                   encoded_train_texts.attention_mask, 
                                                   encoded_train_targets)

    if (require_val_split):
        if (num_labels == 2):
            encoded_val_targets = torch.tensor(target_encoder.transform(val_targets),dtype=torch.long)
        
        elif (num_labels > 2):
            classes = list(target_encoder.categories_[0])
            encoded_val_targets = torch.tensor(target_encoder.transform(np.array(val_targets).reshape(-1,1)).toarray())
        
        val_dataset = torch.utils.data.TensorDataset(encoded_val_texts.input_ids, 
                                                     encoded_val_texts.attention_mask,
                                                     encoded_val_targets)
    
    ############# Creating DataLoader ###########

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    if(require_val_split):
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    ############# Getting optimizer ############

    optimizer_ = utils.getoptimizer(model, optimizer, learning_rate)

    ############ Training Loop #########

    train_lossvsepochs = []
    
    for _epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_dataloader):
            batch = [x.to(device) for x in batch]
            outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
            loss = outputs.loss
            loss.backward()
            optimizer_.step()
            optimizer_.zero_grad()
            model.zero_grad()
            epoch_loss = epoch_loss + loss*BATCH_SIZE
        
        print('The training loss after epoch {} is {}'.format(_epoch, epoch_loss))
        train_lossvsepochs.append(epoch_loss)

        ################ Evaluating after epoch ##############
        
        actual_val, predictions_val = [], []

        if(eval_after_train_epoch and require_val_split):
            model.eval()
            for i,val_batch in enumerate(val_dataloader):
                val_batch = [x.to(device) for x in val_batch]
                
                with torch.no_grad():
                    pred = model(input_ids=val_batch[0], attention_mask=val_batch[1])
                
                logits = pred.logits.detach().cpu().numpy()
                pred_class = list(np.argmax(logits, axis=1))
                pred_class = [classes[x] for x in pred_class]
                
                actual_class = list(target_encoder.inverse_transform(val_batch[2].detach().cpu().numpy()).ravel())
                
                predictions_val.extend(pred_class)
                
                actual_val.extend(actual_class)
  
            f1 = f1_score(actual_val, predictions_val, average=None, labels=uniq_labels)
            print('Val f1 score {}'.format(dict(zip(uniq_labels, list(f1)))))
    
    return(model, tokenizer)        
    
    
    
    
