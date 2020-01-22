library(tictoc)
library(plyr)
library(gplots)
library(sparsefactor)

# source('code/data_simulation.R')
# sourceCpp("code/gibbs.cpp")

set.seed(2020)
data <- simulate.data(K=4, N=100, G=20,
                      pivec=c(rep(0.1, 2), rep(0.9, 2)),
                      alphavec=c(0.5,2,0.5,2))
tic()
samples1 <- gibbs(10000, data$ymat, c(rep(0.1, 2), rep(0.9, 2)),
                  1, 1, 1, 1, seed=2020)
toc()
tic()
samples2 <- gibbs(10000, data$ymat, c(rep(0.5, 2), rep(0.5, 2)),
                  1, 1, 1, 1, seed=2020)
toc()


heatmap.2(aaply(samples1$zmat[1001:10000,,], c(2,3), mean), dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')

heatmap.2(aaply(samples2$zmat[1001:10000,,], c(2,3), mean), dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')

heatmap.2(data$zmat, dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')

aaply(samples2$zmat[1001:10000,,], c(2,3), mean)

zsums <- aaply(samples1$zmat[,,], c(1,3), sum)
trace.plot(zsums[,1], start=1001)
trace.plot(zsums[,2], start=1001)
trace.plot(samples1$lmat[,11,1], samples1$lmat[,11,2], 1001)


trace.plot(samples1$fmat[,1,1], start=1001)

df <- data.frame(lapply(samples1, function(x) I(alply(x, 1))))
