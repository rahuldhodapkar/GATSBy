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
import sklearn.cluster as clust
import sklearn.decomposition
import umap.umap_ as umap
import os

# viz
import matplotlib.pyplot as plt
import seaborn as sns

# pytorch
import torch

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

visium_path = './data/visium/normal_human_prostate'
visium_raw = sc.read_visium(visium_path)

model = torch.load('./calc/graph_attention/model.pickle')

################################################################################
## Perform Dimensionality Reduction of Latent Space
################################################################################

# Generate dimensional reduction and clustering
X = model.latent_embedding.detach().numpy()
clustering = clust.KMeans(n_clusters=2, random_state=42).fit(X)
X_pca = sklearn.decomposition.PCA(n_components=30).fit_transform(X)

X = np.random.random((5,5))
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X) # ***ERROR in umap-learn, requires call on random data
embedding = reducer.fit_transform(X_pca)

# plot points
sns.scatterplot(
    x = embedding[:,0],
    y = embedding[:,1],
    hue = clustering.labels_
)

plt.savefig('./fig/graph_attention/latent_space_umap.png', dpi=120)

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

