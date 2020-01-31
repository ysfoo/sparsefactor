library(tictoc)
library(plyr)
library(gplots)
library(ggplot2)
library(matrixStats)
library(sparsefactor)

# data generation
set.seed(0)
data <- simulate.data(K=4, N=80, G=40,
                      pivec=c(rep(0.2, 2), rep(0.8, 2)),
                      alphavec=rep(1, 4),
                      taushape=1, taurate=0.01)

param.est <- cavi(data$ymat, c(rep(0.2, 2), rep(0.8, 2)), 0.01, 0.01, 0.01, 0.01,
                  max_iter=1000, seed=0)

# heatmaps
heatmap.2(data$zmat, dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')

heatmap.2(data$fmat, dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')
