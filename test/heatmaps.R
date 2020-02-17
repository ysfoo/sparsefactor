library(tictoc)
library(sparsefactor)
library(lattice)
library(viridis)
library(scales)
library(plyr)


S <- 5
D <- 1
K <- S + D
N <- 100
G <- 800
set.seed(1)
data <- simulate.data(K=K, N=N, G=G,
                      zmat = matrix(c(rep(1, 6 * G / 20), rep(0, 14 * G / 20),
                                      rep(1, 1 * G / 20), rep(0, 5 * G / 20), rep(1, 4 * G / 20), rep(0, 10 * G / 20),
                                      rep(0, 1 * G / 20), rep(1, 1 * G / 20), rep(0, 4 * G / 20), rep(1, 1 * G / 20), rep(0, 3 * G / 20), rep(1, 2 * G / 20), rep(0, 8 * G / 20),
                                      rep(0, 2 * G / 20), rep(1, 1 * G / 20), rep(0, 4 * G / 20), rep(1, 1 * G / 20), rep(0, 4 * G / 20), rep(1, 1 * G / 20), rep(0, 7 * G / 20),
                                      rep(0, 3 * G / 20), rep(1, 1 * G / 20), rep(0, 9 * G / 20), rep(1, 1 * G / 20), rep(0, 6 * G / 20),
                                      rep(1, G)),
                                    nrow=G, ncol=K),
                      alphavec=rep(1, K), tauvec=rep(100, G))

png("true_heatmap.png", width=3, height=4.5, units="in", res=1200)
levelplot(t(data$zmat),
          at=seq(0,1,0.05), aspect=2, col.regions=viridis(100),
          ylim=0.5+c(G,0), scales=list(y=list(at=c())),
          main="Z matrix (truth)",
          xlab="Factors",
          ylab=paste(G,"genes"),
          par.settings=list(layout.heights=list(axis.top=0.5)))
dev.off()

param.list <- readRDS("cavi_k6.rds")
RUNS <- length(param.list)
best.elbo <- -Inf
for(s in 1:RUNS) {
    param.est <- param.list[[s]]
    elbo <- tail(param.est$elbo, 1)
    if(elbo > best.elbo) {
        best.elbo <- elbo
        best.seed <- s
    }
}

best.param <- param.list[[best.seed]]
last.idx <- length(best.param$iter)
zmeans <- best.param$zmean[last.idx,,]
png("cavi_k6_heatmap.png", width=3, height=4.5, units="in", res=1200)
levelplot(t(zmeans[,c(order(colSums(zmeans[,1:5]),decreasing=T),6)]),
          at=seq(0,1,0.05), aspect=2, col.regions=viridis(100),
          ylim=0.5+c(G,0), scales=list(y=list(at=c())),
          main="Posterior mean of Z (VI)",
          xlab="Factors",
          ylab=paste(G,"genes"),
          par.settings=list(layout.heights=list(axis.top=0.5)))
dev.off()

rm(param.list)

samples <- readRDS("gibbs_k6.rds")

zmeans <- aaply(samples$zmat, c(2,3), mean)[,c(5,2,4,3,1,6)]
colnames(zmeans) <- 1:6
png("gibbs_k6_heatmap.png", width=3, height=4.5, units="in", res=1200)
levelplot(t(zmeans),
          at=seq(0,1,0.05), aspect=2, col.regions=viridis(100),
          ylim=0.5+c(G,0), scales=list(y=list(at=c())),
          main="Posterior mean of Z (MCMC)",
          xlab="Factors",
          ylab=paste(G,"genes"),
          par.settings=list(layout.heights=list(axis.top=0.5)))
dev.off()
