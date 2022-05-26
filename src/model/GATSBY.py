#!/usr/bin/env python
#
# Core GAT model for GATSBY
#
# @author Rahul Dhodapkar
#

import torch.nn.functional as F
import torch
from torch.nn import Linear, LayerNorm
from torch_geometric.nn import GATConv

class GATSBY(torch.nn.Module):
    def __init__(self, dataset):
        super(GATSBY, self).__init__()

        self.conv1_hid = 512
        self.conv1_heads = 2

        self.conv2_hid = 512
        self.conv2_heads = 2

        self.embed = Linear(
            in_features=dataset.num_features,
            out_features=self.conv1_hid
        )

        self.conv1 = GATConv(
            in_channels=self.conv1_hid,
            out_channels=self.conv1_hid,
            heads=self.conv1_heads,
            dropout=0.6)
        self.attention1 = None
        self.latent_embedding1 = None
        self.ff1 = Linear(
            in_features=self.conv1_hid*self.conv1_heads,
            out_features=self.conv1_hid*self.conv1_heads,
        )
        self.norm1 = LayerNorm([dataset.x.shape[0], self.conv1_hid * self.conv1_heads])

        self.conv2 = GATConv(
            in_channels=self.conv1_hid * self.conv1_heads,
            out_channels=self.conv2_hid * self.conv2_heads,
            heads=self.conv2_heads,
            dropout=0.6,
            concat=False)
        self.attention2 = None
        self.latent_embedding2 = None
        self.ff2 = Linear(
            in_features=self.conv2_hid*self.conv2_heads,
            out_features=self.conv2_hid*self.conv2_heads,
        )
        self.norm2 = LayerNorm([dataset.x.shape[0], self.conv2_hid * self.conv2_heads])

        self.linear = Linear(
            in_features=self.conv2_hid*self.conv2_heads,
            out_features=dataset.num_features
        )


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.embed(x)
        self.latent_embedding1, self.attention1 = self.conv1(x, edge_index, 
            return_attention_weights=True)
        x = self.norm1(torch.add(x.repeat(1, self.conv1_heads), self.latent_embedding1))
        x = self.ff1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        self.latent_embedding2, self.attention2 = self.conv2(x, edge_index,
            return_attention_weights=True)
        x = self.norm2(self.latent_embedding2 + x)
        x = self.ff2(x)
        x = F.elu(self.latent_embedding2)
        # decoding
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.linear(x)
        return x
