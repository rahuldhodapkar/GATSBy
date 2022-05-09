#!/usr/bin/env python
#
# Core GAT model for GATSBY
#
# @author Rahul Dhodapkar
#

import torch.nn.functional as F
import torch
from torch.nn import Linear, LayerNorm
from src.model.GeneAttentionConv import GeneAttentionConv

class GATSBYGene(torch.nn.Module):
    def __init__(self, expression_matrix, num_heads, embed_dim, input_dim):
        super(GATSBYGene, self).__init__()

        self.conv1_embedding = None
        self.conv2_embedding = None
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.input_dim = input_dim

        self.conv = GeneAttentionConv(
            embed_dim=embed_dim,
            num_heads=num_heads,
            input_dim=input_dim
        )
        self.post_conv_linear = Linear(
            in_features=expression_matrix.shape[1] * input_dim,
            out_features=expression_matrix.shape[1] * input_dim
        )
        self.layer_norm = LayerNorm(
            normalized_shape=expression_matrix.shape[1] * input_dim
        )
        self.linear = Linear(
            in_features=expression_matrix.shape[1] * input_dim,
            out_features=expression_matrix.shape[1]
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.layer_norm(
            x + F.dropout(
                self.conv(x, edge_index)
                , p=0.1
                , training=self.training))
        x = self.layer_norm(
            x + F.dropout(
                F.elu(self.post_conv_linear(x))
                , p=0.1
                , training=self.training))
        self.conv1_embedding = x

        x = self.layer_norm(
            x + F.dropout(
                self.conv(x, edge_index)
                , p=0.1
                , training=self.training))
        x = self.layer_norm(
            x + F.dropout(
                F.elu(self.post_conv_linear(x))
                , p=0.1
                , training=self.training))
        self.conv2_embedding = x

        x = self.linear(x)
        return x
