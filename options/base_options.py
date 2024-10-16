import argparse
import os


class BaseOptions:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        ###########################################
        ########Options to run experiments#########
        ###########################################

        
        self.parser.add_argument(
            '--experiment_name',
            type=str,
            default='experiment',
            help='name of the experiment',
            ),
        

        ###########################################
        ##############Options Dataset##############
        ###########################################

        self.parser.add_argument(
            '--root', 
            type=str, 
            default='data/datasets',
            help='path to the folder containing the csv files',
            )
        
        self.parser.add_argument(
            '--filename',
            type=str,
            default='learning.csv',
            help='name of the csv file',
            )
        

        ###########################################
        ###Options to predict unseen datapoints####
        ###########################################


        
        ###########################################
        ##########Options to log results###########
        ###########################################

        
        ###########################################
        #########Smiles columns in dataset#########
        ###########################################

        self.parser.add_argument(
            '--target_variable',
            type=str,
            default=None,
            help='target variable to predict',
            )
        
        self.parser.add_argument(
            '--target_variable_units',
            type=str,
            default=None,
            help='target variable to predict',
            )
        
        self.parser.add_argument(
            '--mol_id_col',
            type=str,
            default=None,
            help='Column containing ID identifier of each entry',
            )
        
        self.parser.add_argument(
            '--mol_cols',
            type=list,
            default=None,
            help='List of columns containing the smiles strings',
            )

        self.parser.add_argument(
            '--graph_features',
            type=dict,
            default=None,
            help='Name of features to be used as graph level features',
            )
        
        self.parser.add_argument(
            '--ohe_graph_feat',
            type=list,
            default=None,
            help='Dictionary containing the features to be one hot encoded',
            )
        

        ###########################################
        ############Training Options GNN###########
        ###########################################

        self.parser.add_argument(
            '--network_name',
            type=str,
            default='GCN',
            help='Name of the network',
            )
        
        self.parser.add_argument(
            '--split_type',
            type=str,
            default='ncv',
            help='Type of split. Allowed values: tvt (train/val/test), cv (cross validation), or ncv (nested cross validation)',
            )
        
        self.parser.add_argument(
            '--pooling',
            type=str,
            default='mean',
            help='Type of pooling',
            )
        
        self.parser.add_argument(
            '--split_method',
            type=str,
            default='stratified',
            help='Method to split the data. Allowed values: random or stratified',
            )
        
        self.parser.add_argument(
            '--test_size',
            type=float,
            default=0.2,
            help='Size of the test set',
            )
        
        self.parser.add_argument(
            '--val_size',
            type=float,
            default=0.2,
            help='Size of the validation set',
            )
        
        self.parser.add_argument(
            '--folds',
            type=int,
            default=10,
            help='Number of folds',
            )
        
        self.parser.add_argument(
            '--n_classes',
            type=int,
            default=1,
            help='Number of classes',
            )
        
        self.parser.add_argument(
            '--n_convolutions',
            type=int,
            default=2,
            help='Number of convolutions',
            )
        
        self.parser.add_argument(
            '--readout_layers',
            type=int,
            default=2,
            help='Number of readout layers',
            )
        
        self.parser.add_argument(
            '--embedding_dim',
            type=int,
            default=64,
            help='Embedding dimension',
            )
        

        
        self.parser.add_argument(
            '--problem_type',
            type=str,
            default='regression',
            help='Type of problem. Allowed values: classification or regression',
            )
        
        self.parser.add_argument(
            '--optimizer',
            type=str,
            default='Adam',
            help='Type of optimizer',
            )
        
        self.parser.add_argument(
            '--lr',
            type=float,
            default=0.01,
            help='Learning rate',
            )
        
        self.parser.add_argument(
            '--early_stopping',
            type=int,
            default=6,
            help='Early stopping',
            )
        
        self.parser.add_argument(
            '--scheduler',
            type=str,
            default='ReduceLROnPlateau',
            help='Type of scheduler',
            )
        
        self.parser.add_argument(
            '--step_size',
            type=int,
            default=7,
            help='Step size for the scheduler',
            )
        
        self.parser.add_argument(
            '--gamma',
            type=float,
            default=0.7,
            help='Factor for the scheduler',
            )
        
        self.parser.add_argument(
            '--min_lr',
            type=float,
            default=1e-08,
            help='Minimum learning rate for the scheduler',
            )
        
        self.parser.add_argument(
            '--batch_size',
            type=int,
            default=40,
            help='Batch size',
            )
        
        self.parser.add_argument(
            '--epochs',
            type=int,
            default=250,
            help='Number of epochs',
            )  

        
        self.parser.add_argument(
            '--global_seed',
            type=int,
            default=20232023,
            help='Global random seed for reproducibility',
            )
        
        self.initialized = True


    def parse(self):
        if not self.initialized:
            self.initialize()
        self._opt = self.parser.parse_args()

        return self._opt
    
    @staticmethod
    def str2bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif value.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
