library(tictoc)
library(plyr)
library(gplots)
library(ggplot2)
library(matrixStats)
library(sparsefactor)

# gibbs
set.seed(0)
data <- simulate.data(K=4, N=80, G=40,
                      pivec=c(rep(0.2, 2), rep(0.8, 2)),
                      alphavec=rep(1, 4),
                      taushape=1, taurate=0.01)
tic()
samples1 <- burn.thin(gibbs(100000, data$ymat, c(rep(0.2, 2), rep(0.8, 2)),
                  1, 1, 1, 1, seed=0), 20000, 1)
toc()
tic()
samples2 <- gibbs(10000, data$ymat, c(rep(0.2, 2), rep(0.8, 2)),
                  1, 1, 1, 1, seed=1)
toc()

# burn + thin
samples1 <- burn.thin(samples1, 10000, 1)

# relabel
tic()
rs1 <- relabel(samples1, TRUE, TRUE)
toc()

# posterior means
rs.zmeans <- aaply(rs1$zmat, c(2,3), mean)
sum(round(rs.zmeans) == data$zmat)
for(i in 1:4) {
    for(j in 1:4) {
        print(sum(round(zmeans)[,i] == data$zmat[,j]))
    }
}

rs.fmeans <- aaply(rs1$fmat, c(2,3), mean)

# number of active elements in column
zsums <- aaply(rs1$zmat[,,], c(1,3), sum)

# trace plots
par(mfrow=c(2,2), mar=c(4,3,1,1))
j <- 2
for(k in 1:4) {
    vec = rs1$lmat[,j,k]
    len.vec <- length(vec)
    plot(0, 0, xlab="iteration", type="n", ylab=paste0("lmat[,",j,",",k,"]"),
         xlim=c(1,2000), ylim=c(min(vec), max(vec)))
    points(vec[40001:42000], type="l")
}

# density plots
par(mfrow=c(2,2), mar=c(4,2.5,1,1))
for(k in 1:4) {
    plot(density(rs1$fmat[,k,3]), main="")
}

# heatmaps
heatmap.2(aaply(rs1$zmat[,,], c(2,3), mean), dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')

heatmap.2(aaply(samples1$zmat[,,], c(2,3), mean), dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')

heatmap.2(data$zmat, dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')

heatmap.2(aaply(rs1$fmat[,,], c(2,3), mean), dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')

heatmap.2(aaply(samples1$fmat[,,], c(2,3), mean), dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')

heatmap.2(data$fmat, dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')

# predictive y
ys <- array(0, c(5000, 20, 100))
for(i in 5001:10000) {
    ys[i-5000,,] <- samples1$lmat[i,,] %*% samples1$fmat[i,,]
}
ymean <- aaply(ys, c(2,3), mean)

# relative error
ds <- abs((ymean - data$ymat) / data$ymat)
data.ds <- abs((data$lmat %*% data$fmat - data$ymat) / data$ymat)

# load samples into dataframe
df <- data.frame(lapply(samples1, function(x) I(alply(x, 1))))
