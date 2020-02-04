library(tictoc)
library(plyr)
library(gplots)
library(ggplot2)
library(matrixStats)
library(sparsefactor)

# two dense two sparse
S <- 2
D <- 2
N <- 100
G <- 200
set.seed(0)
data <- simulate.data(K=S+D, N=N, G=G,
                      zmat = matrix(c(rep(1, G / 5), rep(0, 4 * G / 5),
                                      rep(0, G / 10), rep(1, G / 5), rep(0, 7 * G / 10),
                                      rep(1, G / 20), rep(0, G / 5), rep(1, 3 * G / 4),
                                      rep(0, 3 * G / 20), rep(1, 4 * G / 5), rep(0, G / 20)),
                                    nrow=G, ncol=4),
                      alphavec=rep(1, S+D),
                      tauvec=rep(10, G))

S <- 1
D <- 1
N <- 100
G <- 200
set.seed(0)
data <- simulate.data(K=S+D, N=N, G=G,
                      zmat = matrix(c(rep(1, G / 5),
                                      rep(0, G),
                                      rep(1, 4 * G / 5)), nrow=G, ncol=2),
                      alphavec=rep(1, S+D),
                      tauvec=rep(100, G))

S1 <- 2 # number of sparse1 factors
S2 <- 3 # number of sparse2 factors
D1 <- 2 # number of dense factors
D2 <- 1 # number of dense factors
S <- S1 + S2
D <- D1 + D2
K <- S + D
N <- 40 # number of individuals
G <- 100 # number of genes
set.seed(0)
data <- simulate.data(K=K, N=N, G=G,
                      pivec=c(rep(0.1, S1), rep(0.2, S2), rep(0.8, D1), rep(1, D2)),
                      alphashape=1, alpharate=1,
                      snr=0.1)

tic()
samples <- gibbs(10000, data$ymat, c(rep(0.1, S), rep(0.9, D)),
                 1, 1, 1, 1, burn_in=10000, thin=4, seed=0)
toc()

setul(FALSE)

# relabel
tic()
rs <- relabel(samples, p_every=TRUE, print_cost=TRUE)
toc()

# relabel
tic()
alt <- relabel(samples, p_every=FALSE, print_cost=TRUE)
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
zmeans <- aaply(alt$zmat[,,], c(1,3), mean)
colMeans(zmeans)
colMeans(data$zmat)

# trace plots
par(mfrow=c(4,2), mar=c(4,3,1,1))
j <- 1
for(k in 1:8) {
    vec = alt$fmat[,k,j]
    len.vec <- length(vec)
    plot(0, 0, xlab="iteration", type="n", ylab=paste0("fmat[,",k,",",j,"]"),
         xlim=c(1,len.vec), ylim=c(min(vec), max(vec)))
    points(vec, pch=20, cex=0.5)
}

par(mfrow=c(4,2), mar=c(4,3,1,1))
for(k in 1:8) {
    vec = zmeans[,k]
    len.vec <- length(vec)
    plot(0, 0, xlab="iteration", type="n", ylab=paste0("zsums[,",k,"]"),
         xlim=c(1,len.vec), ylim=c(min(vec), max(vec)))
    points(vec, pch=20, cex=0.5)
}

# density plots
par(mfrow=c(2,2), mar=c(4,2.5,1,1))
for(k in 1:4) {
    plot(density(rs1$fmat[,k,3]), main="")
}

par(mfrow=c(2,1))
lmeans <- aaply(samples$lmat[,,], c(2,3), mean)
plot(lmeans[,1])
points(data$lmat[,2], col="red")
plot(lmeans[,2])
points(data$lmat[,2], col="red")

fmeans <- aaply(samples$fmat[,,], c(2,3), mean)
plot(fmeans[1,])
plot(fmeans[2,])

# heatmaps
heatmap.2(aaply(alt$zmat[,,], c(2,3), mean), dendrogram='none',
          Rowv=FALSE, Colv=FALSE,trace='none')

heatmap.2(aaply(samples$zmat[,,], c(2,3), mean), dendrogram='none',
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
