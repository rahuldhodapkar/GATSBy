#!/usr/bin/env python
#
# Calculate enrichment of receptor/ligand genes with spot attention.
#
# @author Rahul Dhodapkar <rahul.dhodapkar@yale.edu>
#

import scipy.stats as stats
import pandas as pd
import numpy as np

## torch
import torch

## scRNAseq
import scanpy
import anndata

################################################################################
## Load Data
################################################################################

visium_path = './data/visium/normal_human_prostate'
visium_raw = scanpy.read_visium(visium_path)

model = torch.load('./calc/graph_attention/model.pickle')

receptor_ligand_genes = pd.read_csv('./data/gene_input.csv')

################################################################################
## Calculate Source Attention
################################################################################

receptor_ligand_ixs = np.zeros(len(visium_raw.var_names))

for i in range(len(visium_raw.var_names)):
    if visium_raw.var_names[i] in receptor_ligand_genes.gene_name.to_list():
        receptor_ligand_ixs[i] = 1

receptor_ligand_mask = np.array(receptor_ligand_ixs, dtype=bool)

src_att = model.conv2.att_src[0,0,:].detach().numpy()
src_att_rank = src_att.argsort().argsort() # rank
stats.mannwhitneyu(x = src_att[receptor_ligand_mask],
      y = src_att[np.invert(receptor_ligand_mask)])


visium_raw.var_names[src_att.argsort()]


