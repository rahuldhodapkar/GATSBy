#!/usr/bin/env Rscript
#
#

library(ggplot2)
library(LDATS)
library(dplyr)
library(reshape2)
library(viridis)
library(igraph)

gene.gene.attn <- read.csv('./calc/gene_attention/gene_to_gene_attn_df.csv')
rownames(gene.gene.attn) <- gene.gene.attn$X

gene.gene.attn.mat <- gene.gene.attn[,2:ncol(gene.gene.attn)] %>%
                        as.matrix() %>%
                        t() %>%
                        softmax() # row-wise
rownames(gene.gene.attn.mat) <- gene.gene.attn$X
colnames(gene.gene.attn.mat) <- gene.gene.attn$X

plot.df <- melt(gene.gene.attn.mat)
plot.df$Var1 <- factor(plot.df$Var1, levels=gene.gene.attn$X, ordered=T)
plot.df$Var2 <- factor(plot.df$Var2, levels=gene.gene.attn$X, ordered=T)

ggplot(plot.df, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  scale_fill_viridis()

plot.df$weight = plot.df$value
# now look through existing gene modules?
gene.graph <- graph_from_data_frame(plot.df, directed = T)
mod.opt.fast.comms <- cluster_walktrap(gene.graph)