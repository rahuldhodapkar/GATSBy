#!/usr/bin/env Rscript
#
# Calculate graph centrality from igv graphs and project onto visium images
#
# @author Rahul Dhodapkar
#

################################################################################
## Imports
################################################################################

## general
import csv
import igraph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## scRNAseq
import scanpy as sc
import anndata

################################################################################
## Load Data
################################################################################

visium_path = './data/visium/normal_human_prostate'
visium_raw = scanpy.read_visium(visium_path)

importance_df = pd.read_csv('./calc/graph_attention/importance_df.csv')
original_edge_df = pd.read_csv('./calc/graph_attention/original_edge_df.csv')

importance_graph = igraph.Graph(
    edges= [(importance_df.i[x], importance_df.j[x]) 
        for x in range(len(importance_df))]
    ,edge_attrs = {
        'weight': importance_df.v
    }
)
original_graph = igraph.Graph(
    edges = [(original_edge_df.i[x], original_edge_df.j[x]) 
        for x in range(len(original_edge_df))]
    ,edge_attrs = {
        'weight': original_edge_df.v
    }
)

visium_raw.obs['importance_pagerank'] = importance_graph.pagerank(weights='weight')
visium_raw.obs['baseline_pagerank'] = original_graph.pagerank(weights='weight')
visium_raw.obs['pagerank_diff'] = (
    visium_raw.obs['importance_pagerank'] - visium_raw.obs['baseline_pagerank']
)

plt.rcParams["figure.figsize"] = (8, 12)
sc.pl.spatial(
    visium_raw,
    img_key='hires',
    color=['importance_pagerank', 'baseline_pagerank'])

plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(
    visium_raw,
    img_key='hires',
    color=['pagerank_diff'])


