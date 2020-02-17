library(tictoc)
library(sparsefactor)
library(lattice)
library(viridis)
library(scales)
library(plyr)

get.conf.mat <- function(true.z, zmeans) {
    conf.mat <- matrix(nrow=2, ncol=2)
    zpred <- round(zmeans)
    for(i in 0:1) {
        for(j in 0:1) {
            conf.mat[i+1,j+1] <- sum((true.z == i) & (zpred == j))
        }
    }
    return (conf.mat / length(true.z))
}

get.zacc <- function(true.z, zmeans) {
    conf.mat <- get.conf.mat(true.z, zmeans)
    return (conf.mat[1,1] + conf.mat[2,2])
}

get.ferr <- function(true.f, fmeans) {
    for(k in 1:dim(fmeans)[1]) {
        div <- sum(fmeans[k,]^2)
        if(div > 0) {
            fmeans[k,] <- fmeans[k,] * sqrt(sum(true.f[k,]^2) / div)
        }
    }
    return (sqrt(sum((fmeans - true.f)^2) / sum(true.f^2)))
}

get.lferr <- function(true.l, true.f, lmeans, fmeans) {
    true.lf <- true.l %*% true.f
    lfmeans <- lmeans %*% fmeans
    return (sqrt(sum((lfmeans - true.lf)^2) / sum(true.lf^2)))
}

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

samples <- readRDS("gibbs_k6.rds")

time.l <- 0.1
max.len <- length(samples$time)

lsum <- matrix(0, nrow=G, ncol=K)
zsum <- matrix(0, nrow=G, ncol=K)
fsum <- matrix(0, nrow=K, ncol=N)
f2sum <- matrix(0, nrow=K, ncol=N)

times <- rep(0, max.len)
zaccs <- rep(0, max.len)
zprobs <- rep(0, max.len)
ferrs <- rep(0, max.len)
lferrs <- rep(0, max.len)

in.idx <- 0
out.idx <- 0

t <- time.l

while(in.idx < max.len) {
    in.idx <- in.idx + 1

    lsum <- lsum + samples$lmat[in.idx,,]
    zsum <- zsum + samples$zmat[in.idx,,]
    fsum <- fsum + samples$fmat[in.idx,,]
    f2sum <- f2sum + samples$fmat[in.idx,,]^2

    if((samples$time[in.idx] > t) | (in.idx == max.len)) {
        print(in.idx)
        t <- t * 10^0.3
        out.idx <- out.idx + 1
        times[out.idx] <- samples$time[in.idx]

        lmeans <- lsum / in.idx
        fmeans <- fsum / in.idx
        fsigs <- f2sum / in.idx - fmeans^2
        zmeans <- zsum / in.idx
        pz <- matrix(rep(colMeans(zmeans), G), byrow=TRUE, nrow=G)
        data.label <- relabel_truth(data, fmeans, fsigs, pz)
        zaccs[out.idx] <- get.zacc(data.label$zmat, zmeans)
        ferrs[out.idx] <- get.ferr(data.label$fmat, fmeans)
        lferrs[out.idx] <- get.lferr(data.label$lmat, data.label$fmat,
                                     lmeans * zmeans, fmeans)
    }
}

times <- times[1:out.idx]
zaccs <- zaccs[1:out.idx]
ferrs <- ferrs[1:out.idx]
lferrs <- lferrs[1:out.idx]

zprior <- matrix(rep(c(rep(0.1, S), rep(0.9, D)), G), byrow=TRUE, nrow=G, ncol=S+D)
zacc.prior <- get.zacc(data$zmat, zprior)

time.r <- 4000

svg("gibbs_k6_plots.svg")

par(mfrow=c(3,1), mar=c(4.5,4.5,1,1))

plot(c(time.l, time.r), rep(zacc.prior, 2), type="l", pch=19, lwd=2,
     xlab="Time (s)", ylab="Predictive accuracy of Z",
     log="x", xlim=c(time.l, time.r), ylim=c(0.5, 1))
lines(times, zaccs, type="o", pch=19, lwd=2, cex=0.1, col="firebrick")
legend(200, 0.65, legend=c("Prior", "MCMC"),
       col=c("black", "firebrick"), lty=c(1,1), cex=0.8)

plot(1, 1, type="n",
     xlab="Time (s)", ylab="RRMSE of F",
     log="x", xlim=c(time.l, time.r), ylim=c(0, 1.2))
lines(times, ferrs, type="o", pch=19, lwd=2, cex=0.1, col="firebrick")
legend(200, 1.2, legend=c("MCMC"),
       col=c("firebrick"), lty=c(1,1), cex=0.8)

plot(1, 1, type="n",
     xlab="Time (s)", ylab="RRMSE of L*F",
     log="x", xlim=c(time.l, time.r), ylim=c(0, 1.2))
lines(times, lferrs, type="o", pch=19, lwd=2, cex=0.1, col="firebrick")
legend(200, 1.2, legend=c("MCMC"),
       col=c("firebrick"), lty=c(1,1), cex=0.8)

dev.off()


