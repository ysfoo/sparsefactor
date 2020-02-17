library(tictoc)
library(sparsefactor)
library(lattice)
library(viridis)
library(scales)

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

param.list <- readRDS("cavi_k6.rds")

max.time <- 0
time.l <- 0.1
time.r <- 10000

RUNS <- length(param.list)
best.elbo <- -Inf

time.list <- list()
zacc.list <- list()
ferr.list <- list()
lferr.list <- list()
for(s in 1:RUNS) {
    print(s)

    param.est <- param.list[[s]]
    max.len <- length(param.est$iter)
    times <- rep(0, max.len)
    zaccs <- rep(0, max.len)
    zprobs <- rep(0, max.len)
    ferrs <- rep(0, max.len)
    lferrs <- rep(0, max.len)

    end.time <- param.est$time[max.len]
    max.time <- max(max.time, end.time)
    t <- time.l
    in.idx <- 0
    out.idx <- 0
    while(in.idx < max.len) {
        in.idx <- in.idx + 1
        if((param.est$time[in.idx] > t) | (in.idx == max.len)) {
            t <- t * 10^0.1
            out.idx <- out.idx + 1
            times[out.idx] <- param.est$time[in.idx]

            lmeans <- param.est$lmean[in.idx,,]
            fmeans <- param.est$fmean[in.idx,,]
            fsigs <- matrix(rep(diag(param.est$fsig[in.idx,,]), N), nrow=K)
            zmeans <- param.est$zmean[in.idx,,]
            pz <- matrix(rep(colMeans(zmeans), G), byrow=TRUE, nrow=G)
            data.label <- relabel_truth(data, fmeans, fsigs, pz)
            zaccs[out.idx] <- get.zacc(data.label$zmat, zmeans)
            ferrs[out.idx] <- get.ferr(data.label$fmat, fmeans)
            lferrs[out.idx] <- get.lferr(data.label$lmat, data.label$fmat,
                                         lmeans * zmeans, fmeans)
        }
    }
    time.list[[s]] <- times[1:out.idx]
    zacc.list[[s]] <- zaccs[1:out.idx]
    ferr.list[[s]] <- ferrs[1:out.idx]
    lferr.list[[s]] <- lferrs[1:out.idx]

    elbo <- param.est$elbo[out.idx]
    if(elbo > best.elbo) {
        best.elbo <- elbo
        best.seed <- s
    }
}

samples <- readRDS("gibbs_k6.rds")
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

t <- 10

while(in.idx < max.len) {
    in.idx <- in.idx + 1

    lsum <- lsum + samples$lmat[in.idx,,]
    zsum <- zsum + samples$zmat[in.idx,,]
    fsum <- fsum + samples$fmat[in.idx,,]
    f2sum <- f2sum + samples$fmat[in.idx,,]^2

    if(((samples$time[in.idx] > t) & (in.idx > 10)) | (in.idx == max.len)) {
        print(in.idx)
        t <- t * 10^0.1
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

w <- 7
h <- 9

svg("k6_zacc.svg", height=h, width=w)
par(mfrow=c(1,1), mar=c(5,5,4,2), cex.axis=1.2, cex.lab=1.6)

plot(log(c(time.l, time.r), 10), rep(zacc.prior, 2), type="l", pch=19, lwd=2,
     xlab="Log10 time (s)", ylab="Accuracy of Z",
     xlim=log(c(time.l, time.r), 10), ylim=c(0.5, 1))
lines(log(times, 10), zaccs, type="l", pch=19, lwd=2, col="firebrick")
for(s in 1:RUNS) {
    lines(log(time.list[[s]], 10), zacc.list[[s]], type="l", pch=19, lwd=2,
          col=ifelse(s == best.seed, "steelblue", alpha("darkorchid4", 0.4)),
          lty=ifelse(s == best.seed, 1, 2))
}
legend(1, 0.65, legend=c("Prior", "MCMC", "VI (best ELBO)", "VI (other)"),
       col=c("black","firebrick","steelblue",alpha("darkorchid4", 0.4)), lty=c(1,1,1,2), cex=1.6, lwd=2)

svg("k6_ferr.svg", height=h, width=w)
par(mfrow=c(1,1), mar=c(5,5,4,2), cex.axis=1.2, cex.lab=1.6)

plot(1, 1, type="n",
     xlab="Log10 time (s)", ylab="RRMSE of F",
     xlim=log(c(time.l, time.r), 10), ylim=c(0, 1.15))
lines(log(times, 10), ferrs, type="l", pch=19, lwd=2, col="firebrick")
for(s in 1:RUNS) {
    lines(log(time.list[[s]], 10), ferr.list[[s]], type="l", pch=19, lwd=2,
          col=ifelse(s == best.seed, "steelblue", alpha("darkorchid4", 0.4)),
          lty=ifelse(s == best.seed, 1, 2))
}
legend(1, 1.15, legend=c("MCMC", "VI (best ELBO)", "VI (other)"),
       col=c("firebrick","steelblue",alpha("darkorchid4", 0.4)), lty=c(1,1,2), cex=1.6, lwd=2)

svg("k6_lferr.svg", height=h, width=w)
par(mfrow=c(1,1), mar=c(5,5,4,2), cex.axis=1.2, cex.lab=1.6)
plot(1, 1, type="n",
     xlab="Log10 time (s)", ylab="RRMSE of LF",
     xlim=log(c(time.l, time.r), 10), ylim=c(0, 0.21))
lines(log(times, 10), lferrs, type="l", pch=19, lwd=2, col="firebrick")
for(s in 1:RUNS) {
    lines(log(time.list[[s]], 10), lferr.list[[s]], type="l", pch=19, lwd=2,
          col=ifelse(s == best.seed, "steelblue", alpha("darkorchid4", 0.4)),
          lty=ifelse(s == best.seed, 1, 2))
}
legend(1, 0.19, legend=c("MCMC", "VI (best ELBO)", "VI (other)"),
       col=c("firebrick","steelblue",alpha("darkorchid4", 0.4)), lty=c(1,1,2), cex=1.6, lwd=2)

dev.off()

