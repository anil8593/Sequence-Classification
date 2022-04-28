import pandas as pd
import numpy as np
from transformers.models.auto.configuration_auto import AutoConfig
import utils
import os
import torch
import pickle
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def evaluate(args):
    ###### Model Parameters ###
    BATCH_SIZE = args.batch_size
    model_name= args.basemodel_name
    device = 'cuda' if args.use_gpu else 'cpu'
    test_filename = args.test_filename
    do_multilabel_pred = args.multilabel_prediction
    multilabel_pred_threshold = args.multilabel_prob_threshold
    output_probabilities = args.output_all_probabilities
    results_save_as = args.result_save_as

    ###### Reading the test file ######
    df = utils.read_from_csv(os.path.join('input', test_filename))

    ###### Loading the model and tokenizer ######

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.to(device)

    ###### Loading the target encoder ######
    with open(os.path.join(model_name, 'target_encoder.pickle'), 'rb') as f:
        target_encoder = pickle.load(f)
    
    ###### Input encoding ######
    
    encoded_texts = tokenizer.batch_encode_plus(batch_text_or_text_pairs = df['text'].tolist(),
                                                add_special_tokens=True,
                                                padding='max_length',
                                                truncation=True,
                                                is_split_into_words=False,
                                                return_tensors='pt')
    
    ###### Converting to torch datasets and data loader ######

    dataset = torch.utils.data.TensorDataset(encoded_texts.input_ids,
                                             encoded_texts.attention_mask)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    ###### Prediction start ######

    model.eval()

    pred_classes = []
    pred_probs = []

    for batch_ in tqdm(dataloader):
        batch_ = [x.to(device) for x in batch_]

        with torch.no_grad():
            pred = model(input_ids=batch_[0], attention_mask=batch_[1])
        
        logits = pred.logits
        probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()

        if (do_multilabel_pred and multilabel_pred_threshold != 0):
            class_index_ab_threshold, probs_ab_threshold = utils.get_prediction_classes(probs, multilabel_pred_threshold)
            classes_ab_threshold = [list(target_encoder.categories_[0][x]) for x in class_index_ab_threshold]
            pred_classes.extend(classes_ab_threshold)
            pred_probs.extend(probs_ab_threshold)
        
        elif(do_multilabel_pred and multilabel_pred_threshold == 0):
            raise('Please provide a threshold to do multiclass prediction')
        
        elif(output_probabilities):
            multilabel_pred_threshold = 0
            class_index_ab_threshold, probs_ab_threshold = utils.get_prediction_classes(probs, multilabel_pred_threshold)
            classes_ab_threshold = [list(target_encoder.categories_[0][x]) for x in class_index_ab_threshold]
            pred_classes.extend(classes_ab_threshold)
            pred_probs.extend(probs_ab_threshold)
        
        else:
            pred_class = list(np.argmax(probs, axis=1))
            pred_class = [list(target_encoder.categories_[0])[x] for x in pred_class]
            pred_prob = list(np.max(probs, axis=1))
            pred_classes.extend(pred_class)
            pred_probs.extend(pred_prob)
    
    df['Pred'] = pred_classes
    df['Prob'] = pred_probs

    df.to_csv(os.path.join('output', results_save_as), index=None)

    return('Success')

if(__name__ == '__main__'):
    evaluate()


