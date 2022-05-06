#!/usr/bin/env Rscript
#
# Cluster data based on latent space from graph attention.
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
import sklearn.cluster
import sklearn.decomposition
import sklearn.manifold
import os

# viz
import matplotlib.pyplot as plt
import seaborn as sns

# pytorch
#import torch

## scRNAseq
import scanpy as sc
import anndata

################################################################################
## Create Output Scaffolding
################################################################################

os.makedirs('./fig/graph_attention', exist_ok=True)

################################################################################
## Load Data
################################################################################

visium_path = './data/visium/human_prostate_adenocarcinoma'
visium_raw = sc.read_visium(visium_path)

#model = torch.load('./calc/graph_attention/model.pickle')

X = np.load('./calc/graph_attention/embeddings.npy')

################################################################################
## Perform Dimensionality Reduction of Latent Space
################################################################################

# Generate dimensional reduction and clustering
#X = model.latent_embedding2.detach().numpy()
clustering = sklearn.cluster.KMeans(n_clusters=4, random_state=42).fit(X)
#X_pca = sklearn.decomposition.PCA(n_components=30).fit_transform(X)

reducer = sklearn.manifold.TSNE(random_state=42)
embedding = reducer.fit_transform(X) # can also fit to X_pca

# plot points
sns.scatterplot(
    x = embedding[:,0],
    y = embedding[:,1],
    hue = clustering.labels_
)

plt.savefig('./fig/graph_attention/latent_space_tsne.png', dpi=120)

# compare against tsne + pca w/out 
sc.tl.pca(visium_raw, n_comps=30)


################################################################################
## Construct Pseudotime from Latent Space
################################################################################

adata = sc.AnnData(X)
sc.pp.neighbors(adata)
sc.tl.diffmap(adata, n_comps=10)
sc.tl.dpt(adata, n_dcs=10)

################################################################################
## View Clustering on Latent Space
################################################################################

visium_raw.obs['kmeans_clust'] = clustering.labels_
plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(
    visium_raw,
    img_key='hires',
    color=['kmeans_clust'],
    alpha=0.4,
    show=False)
plt.savefig('./fig/graph_attention/tissue_clustering.png', dpi=300)
