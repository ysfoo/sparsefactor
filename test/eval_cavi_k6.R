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
RUNS <- length(param.list)
best.elbo <- -Inf

time.list <- list()
zacc.list <- list()
ferr.list <- list()
lferr.list <- list()
for(s in 1:RUNS) {
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

zprior <- matrix(rep(c(rep(0.1, S), rep(0.9, D)), G), byrow=TRUE, nrow=G, ncol=S+D)
zacc.prior <- get.zacc(data$zmat, zprior)

time.r <- 200

svg("cavi_k6_plots.svg")
par(mfrow=c(3,1), mar=c(4.5,4.5,1,1))
plot(c(time.l, time.r), rep(zacc.prior, 2), type="l", pch=19, lwd=2,
     xlab="Time (s)", ylab="Predictive accuracy of Z",
     log="x", xlim=c(time.l, time.r), ylim=c(0.5, 1))
for(s in 1:RUNS) {
    lines(time.list[[s]], zacc.list[[s]], type="o", pch=19, lwd=2, cex=0.1,
          col=ifelse(s == best.seed, "steelblue", alpha("darkorchid4", 0.2)),
          lty=ifelse(s == best.seed, 1, 2))
}
legend(10, 0.65, legend=c("Prior", "Best ELBO run", "Other runs"),
       col=c("black","steelblue",alpha("darkorchid4", 0.2)), lty=c(1,1,2), cex=0.8)

plot(1, 1, type="n",
     xlab="Time (s)", ylab="RRMSE of F",
     log="x", xlim=c(time.l, time.r), ylim=c(0, 1.2))
for(s in 1:RUNS) {
    lines(time.list[[s]], ferr.list[[s]], type="o", pch=19, lwd=2, cex=0.1,
          col=ifelse(s == best.seed, "steelblue", alpha("darkorchid4", 0.2)),
          lty=ifelse(s == best.seed, 1, 2))
}
legend(10, 1.2, legend=c("Best ELBO run", "Other runs"),
       col=c("steelblue",alpha("darkorchid4", 0.2)), lty=c(1,2), cex=0.8)

plot(1, 1, type="n",
     xlab="Time (s)", ylab="RRMSE of L*F",
     log="x", xlim=c(time.l, time.r), ylim=c(0, 0.3))
for(s in 1:RUNS) {
    lines(time.list[[s]], lferr.list[[s]], type="o", pch=19, lwd=2, cex=0.1,
          col=ifelse(s == best.seed, "steelblue", alpha("darkorchid4", 0.2)),
          lty=ifelse(s == best.seed, 1, 2))
}
legend(10, 0.3, legend=c("Best ELBO run", "Other runs"),
       col=c("steelblue",alpha("darkorchid4", 0.2)), lty=c(1,2), cex=0.8)

dev.off()

