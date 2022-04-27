#!/usr/bin/env python
#
# Core GAT model for GATSBY
#
# @author Rahul Dhodapkar
#

import torch.nn.functional as F
import torch
from torch.nn import Linear
from torch_geometric.nn import GATConv

class GATSBY(torch.nn.Module):
    def __init__(self, dataset):
        super(GATSBY, self).__init__()

        self.conv1_hid = 64
        self.conv1_heads = 8
        self.out_head1 = 1

        self.conv2_hid = 64
        self.conv2_heads = 1

        self.conv1 = GATConv(
            in_channels = dataset.num_features,
            out_channels = self.conv1_hid,
            heads = self.conv1_heads,
            dropout=0.6)
        self.attention1 = None
        self.latent_embedding1 = None

        self.conv2 = GATConv(
            in_channels = self.conv1_hid*self.conv1_heads,
            out_channels = self.conv2_hid,
            heads=self.conv2_heads,
            dropout=0.6,
            concat=False)
        self.attention2 = None
        self.latent_embedding2 = None

        self.linear = Linear(
            in_features = self.conv2_hid*self.conv2_heads,
            out_features = dataset.num_features
        )


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        self.latent_embedding1, self.attention1 = self.conv1(x, edge_index, 
            return_attention_weights=True)
        x = F.elu(self.latent_embedding1)
        x = F.dropout(x, p=0.6, training=self.training)
        self.latent_embedding2, self.attention2 = self.conv2(x, edge_index,
            return_attention_weights=True)
        x = F.elu(self.latent_embedding2)
        # decoding
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.linear(x)
        return x
