#!/usr/bin/env python
## GeneAttentionConv.py
#
# Gene Attention Convolutional Layer
#
# @author Rahul Dhodapkar
#

import torch
from torch.nn import Sequential as Seq, Linear, ReLU
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class GeneAttentionConv(MessagePassing):
    def __init__(self, embed_dim, num_heads, input_dim):
        super().__init__(aggr='mean')

        self.attn_output_weights = None
        self.attn_output = None

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True)
        self.input_dim = input_dim

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        if x_i.shape[1] % self.input_dim != 0:
            print("Error, invalid configuration")

        x_i_reshape = torch.reshape(x_i, (x_i.shape[0], int(x_i.shape[1] / self.input_dim), self.input_dim))
        x_j_reshape = torch.reshape(x_j, (x_j.shape[0], int(x_j.shape[1] / self.input_dim), self.input_dim))

        self.attn_output, self.attn_output_weights = self.multihead_attn(x_i_reshape, x_j_reshape, x_j_reshape)

        output_reshape = torch.reshape(self.attn_output, (x_i.shape[0], x_i.shape[1]))

        return output_reshape
