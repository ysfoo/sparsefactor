library(tictoc)
library(plyr)
library(gplots)
library(ggplot2)
library(matrixStats)
library(sparsefactor)

# data generation

# two dense
S <- 0
D <- 2
N <- 100
G <- 500
set.seed(0)
data <- simulate.data(K=S+D, N=N, G=G,
                      zmat = matrix(c(rep(1, 9 * G / 10), rep(0, 2 * G / 10), rep(1, 9 * G / 10)), nrow=G, ncol=2),
                      alphavec=rep(1, S+D),
                      tauvec=rep(100, G))

# random
set.seed(0)
data <- simulate.data(K=S+D, N=N, G=G,
                      pivec=c(rep(0.2, S), rep(0.8, D)),
                      alphavec=rep(1, S+D),
                      tauvec=rep(100, G))

# one dense one sparse
S <- 1
D <- 1
N <- 100
G <- 500
set.seed(0)
data <- simulate.data(K=S+D, N=N, G=G,
                      zmat = matrix(c(rep(1, G / 5), rep(0, G), rep(1, 4 * G / 5)), nrow=G, ncol=2),
                      alphavec=rep(1, S+D),
                      tauvec=rep(100, G))

# one dense one sparse with overlap
S <- 1
D <- 1
N <- 100
G <- 500
set.seed(0)
data <- simulate.data(K=S+D, N=N, G=G,
                      zmat = matrix(c(rep(1, G / 5),
                                      rep(0, 9 * G / 10),
                                      rep(1, 9 * G / 10)), nrow=G, ncol=2),
                      alphavec=rep(1, S+D),
                      tauvec=rep(100, G))

# two dense two sparse
S <- 2
D <- 2
N <- 100
G <- 800
set.seed(0)
data <- simulate.data(K=S+D, N=N, G=G,
                      zmat = matrix(c(rep(1, G / 5), rep(0, 4 * G / 5),
                                      rep(0, G / 10), rep(1, G / 5), rep(0, 7 * G / 10),
                                      rep(1, G / 20), rep(0, G / 5), rep(1, 3 * G / 4),
                                      rep(0, 3 * G / 20), rep(1, 4 * G / 5), rep(0, G / 20)),
                                    nrow=G, ncol=4),
                      alphavec=rep(1, S+D),
                      tauvec=rep(100, G))

tic()
for(s in 1:50) {
    print(s)
    param.est <- cavi(data$ymat, c(rep(0.2, S), rep(0.8, D)), 0.001, 0.001, 0.001, 0.001,
                      max_iter=1000, seed=s)
    heatmap.2(param.est$zmean, dendrogram='none',
              Rowv=FALSE, Colv=FALSE,trace='none')
    # Sys.sleep(0.1)
}
toc()

param.est <- cavi(data$ymat, c(rep(0.2, S), rep(0.8, D)), 0.01, 0.01, 1, 1,
                  max_iter=1000, seed=7)
heatmap.2(param.est$zmean, dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')

param.est$zmean
param.est$taurate / param.est$taushape
param.est$alpharate / param.est$alphashape

sum((data$zmat[,1] == 1) & (round(param.est$zmean[,3]) == 1))

plot(data$lmat[,2])
points(1.9 * param.est$lmean[,1], col='red')

plot(data$lmat[,1])
points(1.2 * param.est$lmean[,2], col='red')

plot(data$fmat[2,])
points(param.est$fmean[2,] / -1.8, col='red')

# heatmaps
heatmap.2(data$zmat, dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')

heatmap.2(param.est$zmean, dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')

heatmap.2(data$fmat, dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')
