import torch
import torch.nn as nn
from model.networks import BaseNetwork
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp, GATv2Conv
import argparse



class GAT(BaseNetwork):

    def __init__(self, opt: argparse.Namespace, n_node_features:int, pooling:str):
        super().__init__(opt=opt, n_node_features=n_node_features, pooling=pooling)

        self._name = "GCN"

        #First convolution and activation function
        self.linear = nn.Linear(self.n_node_features,
                             self.embedding_dim,)
        self.relu1 = nn.LeakyReLU()


        #Convolutions
        self.conv_layers = nn.ModuleList([])
        for _ in range(self.n_convolutions):
            self.conv_layers.append(GATv2Conv(self.embedding_dim,
                                              self.embedding_dim,
                                              )
                                    )


        #graph embedding is the concatenation of the global mean and max pooling, thus 2*embedding_dim
        graph_embedding = self.graph_embedding

        #Readout layers
        self.readout = nn.ModuleList([])

        for _ in range(self.readout_layers-1):
            reduced_dim = int(graph_embedding/2)
            self.readout.append(nn.Sequential(nn.Linear(graph_embedding, reduced_dim), 
                                              nn.LeakyReLU()))
            graph_embedding = reduced_dim

        #Final readout layer
        self.readout.append(nn.Linear(graph_embedding, self.n_classes))
        
        
        self._make_loss()
        self._make_optimizer(opt.optimizer, opt.lr)
        self._make_scheduler(scheduler=opt.scheduler, step_size = opt.step_size, gamma = opt.gamma, min_lr=opt.min_lr)


    def forward(self,x=None, edge_index=None, batch_index=None, return_graph_embedding=False):


        x = self.linear(x)
        x = self.relu1(x)

        for i in range(self.n_convolutions):
            x = self.conv_layers[i](x, edge_index)
            x = nn.LeakyReLU()(x)

        if self.pooling == 'mean':
            x = gap(x, batch_index)
        elif self.pooling == 'max':
            x = gmp(x, batch_index)
        elif self.pooling == 'sum':
            x = gsp(x, batch_index)
        elif self.pooling == 'mean/max':
            x = torch.cat([gmp(x, batch_index), 
                                gap(x, batch_index)], dim=1)
        
        graph_emb = x
        
        for i in range(self.readout_layers):
            x = self.readout[i](x)

        if self.n_classes == 1:
            x = x.float().squeeze()

        if not return_graph_embedding:
            return x
        else:
            return x, graph_emb