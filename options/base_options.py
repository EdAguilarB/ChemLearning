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
            '--train_GNN', 
            type=self.str2bool,
            nargs='?', 
            const=True, 
            default=True, 
            help='Whether to train the GNN or not'
            )
        
        self.parser.add_argument(
            '--train_tml', 
            type=self.str2bool, 
            nargs='?', 
            const=True, 
            default=True, 
            help='Whether to train the TML or not'
            )
        
        self.parser.add_argument(
            '--compare_models', 
            type=self.str2bool, 
            nargs='?', 
            const=True, 
            default=True, 
            help='Whether to compare GNN and TML or not'
            )
        
        self.parser.add_argument(
            '--predict_unseen', 
            type=self.str2bool, 
            nargs='?', 
            const=True, 
            default=True, 
            help='Whether to predict the unseen data or not'
            )
        

        self.parser.add_argument(
            '--shapley_analysis', 
            type=self.str2bool, 
            nargs='?', 
            const=True, 
            default=True, 
            help='Whether to explain the GNN or not'
            )

        self.parser.add_argument(
            '--denoise_graph', 
            type=self.str2bool, 
            nargs='?', 
            const=True, 
            default=True, 
            help='Whether to explain the GNN or not'
            )

        self.parser.add_argument(
            '--GNNExplainer', 
            type=self.str2bool, 
            nargs='?', 
            const=True, 
            default=True, 
            help='Whether to explain the GNN or not'
            )
        
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
            default='data/datasets/rhcaa_learning',
            help='path to the folder containing the csv files',
            )
        
        self.parser.add_argument(
            '--filename',
            type=str,
            default='learning.csv',
            help='name of the csv file',
            )
        
        self.parser.add_argument(
            '--filename_final_test',
            type=str,
            default='final_test.csv',
            help='name of the csv file for the final test',
            )
        
        self.parser.add_argument(
            '--root_final_test', 
            type=str, 
            default='data/datasets/rhcaa_final_test',
            help='path to the folder containing the csv files',
            )
        
        ###########################################
        ###Options to predict unseen datapoints####
        ###########################################

        self.parser.add_argument(
            '--root_predict', 
            type=str, 
            default='data/datasets/rhcaa_final_test',
            help='path to the folder containing the csv files',
            )
        
        self.parser.add_argument(
            '--filename_predict',
            type=str,
            default='final_test.csv',
            help='name of the csv file for the final test',
            )
        
        
        self.parser.add_argument(
            '--predict_model',
            type=str,
            default='learning_set',
            help='Model to use for prediction. Options: learning_set or all_data',
            )
        
        ###########################################
        ##########Options to log results###########
        ###########################################
        self.parser.add_argument(
            '--log_dir_results',
            type=str,
            default=os.path.join(os.getcwd(), 'results/'),
            help='path to the folder where the results will be saved',
            )
        
        ###########################################
        #########Smiles columns in dataset#########
        ###########################################

        self.parser.add_argument(
            '--target_variable',
            type=str,
            default='%top',
            help='target variable to predict',
            )
        
        self.parser.add_argument(
            '--target_variable_units',
            type=str,
            default='kJ/mol',
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
            type=str,
            default=['Ligand', 'substrate', 'boron reagent'],
            help='column names of the reactant and product smiles',
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
            '--improved',
            type=bool,
            default=True,
            help='Whether to use the improved version of the GCN',
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

        ###########################################
        ############Training Options TML###########
        ###########################################
        
        self.parser.add_argument(
            '--tml_algorithm',
            type=str,
            default='gb',
            help='Traditional ML algorithm to use. Allowed values: lr for linear regression, gb for gradient boosting, or rf for random forest.',
            )

        ###########################################
        ############Explain Options GNN###########
        ###########################################

        # model to be explained
        self.parser.add_argument(
            '--explain_model',
            type=list,
            default=[8, 10],
            help='List of outer, inner fold to explain',
        )

        ######################
        ###Denoise function###
        ######################

        # reaction to denoise 
        self.parser.add_argument(
            '--denoise_reactions',
            type=int,
            default=1,
            help='Reaction to be denoised using the GNNExplainer Algorithm',
        )

        # Denoise the grpah based on a node feature
        self.parser.add_argument(
            '--denoise_based_on',
            type=str,
            default=None,
            help='Denoise the graph based on a given node features. Allowed values: None, atom_identity, degree, hyb, aromatic, ring, chiral, conf',
        )

        # Select molecule to be denoised
        self.parser.add_argument(
            '--denoise_mol',
            type=str,
            default='ligand',
            help='Denoise the given molecule of the reaction. Allowed values: ligand, substrate, boron',
        )

        # wheter or not to normalize the GNNExplainer attribution scores
        self.parser.add_argument(
            '--norm_denoise',
            type=self.str2bool,
            default=False,
            help='Whether or not to normalise the masks per molecule',
        )

        ######################
        #####Shap function####
        ######################
        
        self.parser.add_argument(
            '--shap_index',
            type=list,
            default=[82, 416, 89, 91, 95, 97],
            help='List of index of reactions to explain',
        )

        # molecule to apply shap analysis
        self.parser.add_argument(
            '--shap_mol',
            type=list,
            default='ligand',
            help='Molecule to analyze using shap. Allowed values: "l" for ligand, "s" for substrate, or "b" for organoboron reagent',
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
