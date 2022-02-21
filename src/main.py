import os
import argparse
from train import start_training

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--multiple_train_files', 
                        action="store_true", 
                        help="Add this argument if there are training files are multiple txt \
                             files and class is the folder name. If not specified train file is considered in csv format")
    
    parser.add_argument('--train_filename',
                        default='',
                        type=str,
                        required=False,
                        help="The name of the trainingfile. Should be used only if flag multiple_train_files is not set.\
                              The csv should contain two columns with names \"text\" and \"label\"")
    
    parser.add_argument('--basemodel_name',
                        default='distillbert',
                        type=str,
                        required=True,
                        help="The name of the transformer model to use. \
                              For using custom model provide the path to the custom model.")
    
    parser.add_argument('--tune_basemodel',
                        action='store_true',
                        help='Add this argument if you wish to finetune the weights of the basemodel')
    
    parser.add_argument('--model_save_as',
                        type=str,
                        required=True,
                        help='The name of the model to save with')
    
    parser.add_argument('--use_gpu',
                        action='store_true',
                        help='Add this argument if you have a gpu and wish to use it')
    
    parser.add_argument('--do_train',
                        action='store_true',
                        help='Add this argument if you wish to train the model')
    
    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        required=False,
                        help='The number of training epochs to be used')
    
    parser.add_argument('--batch_size',
                        default=64,
                        type=int,
                        required=False,
                        help='The batch size to be used')
    
    parser.add_argument('--optimiser',
                        default ='adam',
                        type=str,
                        required=False,
                        help='The optimizer to be used for training. Can only take values sgd, adam')
    
    parser.add_argument('--lr',
                        default=0.00001,
                        type=float,
                        required=False,
                        help='The learning rate to be used')

    parser.add_argument('--require_val_split',
                        action='store_true',
                        help='If train valid split required from the train data')
    
    parser.add_argument('--val_fraction',
                        default=0.2,
                        type=float,
                        required=False,
                        help="The fraction of training set to be considered for validation, Applicable only\
                              if require_val_split is provided.")
    
    parser.add_argument('--eval_after_train_epoch',
                        action='store_true',
                        help='This flags enables validation after every training epoch')
    

    args = parser.parse_args()

    if (args.do_train):
        trained_model, tokenizer = start_training(args)

        ######## Saving the model ##########
        trained_model.save_pretrained(os.path.join('model', args.model_save_as))
        tokenizer.save_pretrained(os.path.join('model', args.model_save_as))


if (__name__ =='__main__'):
    main()   