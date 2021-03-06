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

X_pca = sklearn.decomposition.PCA(n_components=30).fit_transform(X)
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_pca)

#reducer = sklearn.manifold.TSNE(random_state=42)
#embedding = reducer.fit_transform(X) # can also fit to X_pca

transcript_graph = sklearn.neighbors.kneighbors_graph(X_pca, n_neighbors=10)
clustering = Louvain(resolution=0.3).fit(transcript_graph)

#clustering = sklearn.cluster.KMeans(n_clusters=2, random_state=42).fit(embedding)

# plot points
sns.scatterplot(
    x=embedding[:,0],
    y=embedding[:,1],
    hue=clustering.labels_
)

plt.savefig('./fig/graph_attention/latent_space_umap.png', dpi=120)

'''
# compare against tsne + pca w/out 
sc.tl.pca(visium_raw, n_comps=30)
plt.savefig('./fig/graph_attention/original_space_tsne.png', dpi=120)
'''

################################################################################
## Construct Pseudotime from Latent Space
################################################################################

'''
adata = sc.AnnData(X)

adata.uns['iroot'] = np.argsort(embedding[:,0])[len(embedding[:,0])-1]

sc.pp.neighbors(adata)
sc.tl.diffmap(adata, n_comps=10)
sc.tl.dpt(adata, n_branchings=0, n_dcs=10, min_group_size=0.01, allow_kendall_tau_shift=True)
#sc.tl.dpt(adata)

sc.pl.dpt(adata)
sc.pl.dpt_groups_pseudotime(adata)
plt.savefig('./fig/graph_attention/latent_space_tsne.png', dpi=120)
'''

################################################################################
## View Clustering on Latent Space
################################################################################

visium_raw.obs['kmeans_clust'] = [str(x) for x in clustering.labels_]
plt.rcParams["figure.figsize"] = (8, 8)
sc.pl.spatial(
    visium_raw,
    img_key='hires',
    color=['kmeans_clust'],
    alpha=1,
    show=False)
plt.savefig('./fig/graph_attention/tissue_clustering_UMAP.png', dpi=300)
