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
from sknetwork.clustering import Louvain

# viz
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

# pytorch
#import torch

## scRNAseq
import scanpy as sc
import anndata

################################################################################
## Create Output Scaffolding
################################################################################

os.makedirs('./fig/standard_spatial_analysis', exist_ok=True)

################################################################################
## Load Data
################################################################################

visium_path = './data/visium/human_prostate_adenocarcinoma'
visium_raw = sc.read_visium(visium_path)

X = np.log1p(visium_raw.X.todense())

################################################################################
## Perform Dimensionality Reduction of Latent Space
################################################################################

X_pca = sklearn.decomposition.PCA(n_components=30).fit_transform(X)
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_pca)

transcript_graph = sklearn.neighbors.kneighbors_graph(X_pca, n_neighbors=10)
clustering = Louvain(resolution=0.3).fit(transcript_graph)

# plot points
sns.scatterplot(
    x=embedding[:,0],
    y=embedding[:,1],
    hue=clustering.labels_
)

plt.savefig('./fig/standard_spatial_analysis/spot_expression_louvain_umap.png', dpi=120)

#############
## View Clustering on Latent Space in Tissue
#############

visium_raw.obs['louvain_clust'] = [str(x) for x in clustering.labels_]
plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(
    visium_raw,
    img_key='hires',
    color=['louvain_clust'],
    alpha=1,
    show=False)
plt.savefig('./fig/standard_spatial_analysis/tissue_louvain_umap.png', dpi=300)

################################################################################
## KMeans K=2
################################################################################

X_pca = sklearn.decomposition.PCA(n_components=30).fit_transform(X)
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_pca)

clustering = sklearn.cluster.KMeans(n_clusters=2, random_state=42).fit(X_pca)

# plot points
sns.scatterplot(
    x=embedding[:,0],
    y=embedding[:,1],
    hue=clustering.labels_
)

plt.savefig('./fig/standard_spatial_analysis/spot_expression_kmeans_umap.png', dpi=120)

#############
## View Clustering on Latent Space in Tissue
#############

visium_raw.obs['kmeans_clust'] = [str((x + 1) % 2) for x in clustering.labels_]
plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(
    visium_raw,
    img_key='hires',
    color=['kmeans_clust'],
    alpha=1,
    show=False)
plt.savefig('./fig/standard_spatial_analysis/tissue_kmeans_umap.png', dpi=300)

