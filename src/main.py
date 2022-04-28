import os
import argparse
import pickle
from train import start_training
from eval import evaluate

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
    
    parser.add_argument('--test_filename',
                        default='',
                        type=str,
                        required=False,
                        help="The name of the testing file. Should be used only if flag multiple_train_files is not set.\
                              The csv should contain two columns with names \"text\". The file would be read from input folder")
    
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
                        default='test_model',
                        type=str,
                        required=False,
                        help='The name of the model to save with')
    
    parser.add_argument('--use_gpu',
                        action='store_true',
                        help='Add this argument if you have a gpu and wish to use it')
    
    parser.add_argument('--do_train',
                        action='store_true',
                        help='Add this argument if you wish to train the model')
    
    parser.add_argument('--do_eval',
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
    
    parser.add_argument('--use_scheduler',
                        action='store_true',
                        help='Add this argument if you wish to activate the scheduler')

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
    
    parser.add_argument('--multilabel_prediction',
                        action='store_true',
                        help='This flags enables multilabel prediction during evaluation. Only considered with do_eval flag\
                             Setting this flag to true required=s setting multiclass_prob_threshold flag too')
    
    parser.add_argument('--multilabel_prob_threshold',
                        default=0.0,
                        type=float,
                        required=False,
                        help="The Probability value above which all classes will be predicted. Should be used with multilcass prediction flag")
    
    parser.add_argument('--output_all_probabilities',
                        action='store_true',
                        help='If set to true this would enable the model to output the probabilities of all classes. Donot use this with multilabel_prediction flag')
                
    parser.add_argument('--result_save_as',
                        default = 'result.csv',
                        type=str,
                        required=False,
                        help='The name of the results file. Results will be stored under the output folder with the provided name')

    args = parser.parse_args()

    if (args.do_train):
        trained_model, tokenizer, target_encoder = start_training(args)

        ######## Saving the model ##########
        trained_model.save_pretrained(os.path.join('model', args.model_save_as))
        tokenizer.save_pretrained(os.path.join('model', args.model_save_as))

        ######### Saving the target encoder #############
        with open(os.path.join('model', args.model_save_as, 'target_encoder.pickle'), 'wb') as f:
            pickle.dump(target_encoder, f)

    if (args.do_eval):
        evaluate(args)

if (__name__ =='__main__'):
    main()   