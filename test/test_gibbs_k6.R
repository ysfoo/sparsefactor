library(tictoc)
library(sparsefactor)
library(lattice)
library(viridis)
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

tic()
samples <- gibbs(20000, data$ymat, c(rep(0.1, S), rep(0.9, D)),
                 1, 1, 1, 1, thin=10, burn_in=1000, seed=1)
toc()
saveRDS(samples, "gibbs_k6.rds")

# tic()
# samples.p0.l0 <- relabel(samples, p_every=FALSE, use_l=FALSE)
# toc()
# saveRDS(samples.p0.l0, "gibbs_k6_p0_l0.rds")

png("gibbs_k6_zprop_trace.png", width=8, height=8, units="in", res=300)
ztrace <- aaply(samples$zmat[,,], c(1,3), mean)
par(mfrow=c(3,2), mar=c(4,3,1,1))
for(k in 1:6) {
    vec = ztrace[,k]
    len.vec <- length(vec)
    plot(0, 0, xlab="iteration", type="n", ylab=paste0("zsums[,",k,"]"),
         xlim=c(1,len.vec), ylim=c(min(vec), max(vec)))
    points(vec, pch=20, cex=0.5)
}
dev.off()

for(j in 1:4) {
    png(paste0("gibbs_k6_f", j, "_trace.png"), width=8, height=8, units="in", res=300)
    par(mfrow=c(3,2), mar=c(4,3,1,1))
    for(k in 1:6) {
        vec = samples$fmat[,k,j]
        len.vec <- length(vec)
        plot(0, 0, xlab="iteration", type="n", ylab=paste0("fmat[,",k,",",j,"]"),
             xlim=c(1,len.vec), ylim=c(min(vec), max(vec)))
        points(vec, pch=20, cex=0.5)
    }
    dev.off()
}
