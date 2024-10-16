import argparse
from copy import copy, deepcopy
from torch_geometric.loader import DataLoader





def make_network(network_name: str, opt: argparse.Namespace, n_node_features: int, pooling: str):
    if network_name == "GCN":
        from model.gcn import GCN
        return GCN(opt=opt, n_node_features=n_node_features, pooling=pooling)
    elif network_name == "GAT":
        from model.gat import GAT
        return GAT(opt=opt, n_node_features=n_node_features, pooling=pooling)
    elif network_name == "GraphSAGE":
        from model.graphsage import GraphSAGE
        return GraphSAGE(opt=opt, n_node_features=n_node_features, pooling=pooling)
    elif network_name == "ChebNet":
        from model.chebnet import ChebNet
        return ChebNet(opt=opt, n_node_features=n_node_features, pooling=pooling)
    else:
        raise ValueError(f"Network {network_name} not implemented")
    

def create_loaders(dataset, opt: argparse.Namespace):

    """
    Creates training, validation and testing loaders for cross validation and
    inner cross validation training-evaluation processes.
    Args:
    dataset: pytorch geometric dataset
    batch_size (int): size of batches
    val (bool): whether or not to create a validation set
    folds (int): number of folds to be used
    num points (int): number of points to use for the training and evaluation process

    Returns:
    (tuple): DataLoaders for training, validation and test set

    """

    batch_size = opt.batch_size
    set = dataset.set

    if opt.split_type == 'tvt':

        test_indices = [i for i, s in enumerate(set) if s == 0]
        val_indices = [i for i, s in enumerate(set) if s == 1]
        train_indices = [i for i, s in enumerate(set) if s != 0 and s != 1]
        
        test_loader = DataLoader(dataset[test_indices], batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(dataset[val_indices], batch_size=batch_size, shuffle=False)
        train_loader = DataLoader(dataset[train_indices], batch_size=batch_size, shuffle=True)

        yield deepcopy((train_loader, val_loader, test_loader))

    elif opt.split_type == 'cv':

        unique_folds = sorted(set.unique())
        print(unique_folds)
        num_folds = len(unique_folds)
        print(num_folds)

        for outer in range(1):

            # outer or test set will always be the first fold: fold 0
            test_indices = [i for i, s in enumerate(set) if s == outer]
            test_loader = DataLoader(dataset[test_indices], batch_size=batch_size, shuffle=False)

            # iterate over the rest of the folds from 1 to n-1
            for inner in range(1, num_folds):  

                # val set is the inner fold
                val_indices = [i for i, s in enumerate(set) if s == inner]
                val_loader = DataLoader(dataset[val_indices], batch_size=batch_size, shuffle=False)

                train_indices = [i for i, s in enumerate(set) if s != outer and s != inner]
                train_loader = DataLoader(dataset[train_indices], batch_size=batch_size, shuffle=True)

                yield deepcopy((train_loader, val_loader, test_loader))


    elif opt.split_type == 'ncv':
        folds = [[] for _ in range(folds)]
        for data in dataset:
            folds[data.fold].append(data)

        for outer in range(len(folds)):
            proxy = copy(folds)
            test_loader = DataLoader(proxy.pop(outer), batch_size=batch_size, shuffle=False)
            for inner in range(len(proxy)):  # length is reduced by 1 here
                proxy2 = copy(proxy)
                val_loader = DataLoader(proxy2.pop(inner), batch_size=batch_size, shuffle=False)
                flatten_training = [item for sublist in proxy2 for item in sublist]  # flatten list of lists
                train_loader = DataLoader(flatten_training, batch_size=batch_size, shuffle=True)
                yield deepcopy((train_loader, val_loader, test_loader))