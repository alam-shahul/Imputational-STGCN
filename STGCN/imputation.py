import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import stgcn
from stgcn import STGCN
# from torch_geometric.nn import GATConv
# from torch_geometric.data import Data

class ImputationalSTGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, attention_embedding_dim=20, num_heads=1):
        """
        :param num_nodes_complete: Number of nodes in the graph.
        :param num_nodes_incomplete: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(ImputationalSTGCN, self).__init__()
        self.complete_predictor = STGCN(num_nodes,
            num_features,
            num_timesteps_input,
            num_timesteps_output)
        
        # self.mapping_matrix = nn.Parameter(torch.FloatTensor(num_nodes_incomplete, num_nodes_complete))
        # self.mapping_matrix = nn.Linear(num_nodes_incomplete,
        #                        num_nodes_complete)
        
        # self.weight_mapper = GATConv(num_features, dataset.num_classes)
#         self.query_weights = nn.Parameter(torch.FloatTensor(num_features * num_timesteps_input, attention_embedding_dim))
#         self.key_weights = nn.Parameter(torch.FloatTensor(num_features * num_timesteps_input, attention_embedding_dim))
#         self.weight_mapper = nn.MultiheadAttention(attention_embedding_dim, num_heads, batch_first=True)
        self.weight_mapper = nn.MultiheadAttention(num_features * num_timesteps_input, num_heads, batch_first=True)
        
    def forward(self, A_hat, X, incomplete_subset=None, incomplete_subset_A_hat=None):
        """
        :param X: Input data of shape (batch_size_complete, num_nodes_incomplete, num_timesteps,
        num_features=in_channels).
        :param incomplete_subset: Input data of shape (batch_size_incomplete, num_nodes_incomplete, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        
        out_complete = self.complete_predictor(A_hat, X)
        
        batch_size_complete, num_nodes_complete, num_timesteps, num_features = X.shape
        flattened_complete_subset = X.view(batch_size_complete, num_nodes_complete, num_features * num_timesteps)
        complete_query = flattened_complete_subset # @ self.query_weights
        complete_key = flattened_complete_subset # @ self.key_weights
        _, complete_attn_weights = self.weight_mapper(complete_query, complete_key, complete_key)

#         print(f"complete_query: {complete_query[0]}")
#         print(f"complete_key: {complete_key[0]}")
#         for n, p in self.weight_mapper.named_parameters():
#             print(n, p)
#             print(n, p.grad)
            
        if incomplete_subset is None:
            out_incomplete = out_complete
            incomplete_attn_output, incomplete_attn_weights = complete_attn_output, complete_attn_weights
        else:
            batch_size_incomplete, num_nodes_incomplete, num_timesteps, num_features = X.shape
            flattened_incomplete_subset = incomplete_subset.view(batch_size_incomplete, num_nodes_incomplete, num_features * num_timesteps)
            incomplete_query = flattened_incomplete_subset # @ self.query_weights
            _, incomplete_attn_weights = self.weight_mapper(incomplete_query, complete_key, complete_key)
            
            out_incomplete = self.complete_predictor(incomplete_subset_A_hat, incomplete_subset)
       
#         print(f"incomplete_attn_weights: {incomplete_attn_weights[0]}")
#         print(f"complete_attn_weights: {complete_attn_weights[0]}")
#         print(f"incomplete_attn_output: {incomplete_attn_output[0]}")
#         print(f"complete_attn_output: {complete_attn_output[0]}")
        
        return out_complete, complete_attn_weights, out_incomplete, incomplete_attn_weights