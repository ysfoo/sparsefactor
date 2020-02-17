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

RUNS <- 10
best.elbo <- -Inf
param.list <- list()
for(s in 1:RUNS) {
    tic()
    param.est <- cavi(data$ymat, c(rep(0.1, S), rep(0.9, D)), 0.001, 0.001, 0.001, 0.001,
                      max_iter=50000, tol=1e-14, check=20, save=20, seed=s)
    toc()
    param.list[[s]] <- param.est
    elbo <- tail(param.est$elbo, 1)
    if(elbo > best.elbo) {
        best.elbo <- elbo
        best.seed <- s
    }
}

saveRDS(param.list, "cavi_k6.rds")

best.param <- param.list[[best.seed]]
last.idx <- length(best.param$iter)

png("cavi_k6_best.png", width=8, height=8, units="in", res=300)
levelplot(t(best.param$zmean[last.idx,,]), at=seq(0,1,0.05), aspect=2, col.regions=viridis(100),
          ylim=0.5+c(G,0), scales=list(y=list(at=c())),
          main="Posterior mean of Z (VI)",
          xlab="Factors",
          ylab=paste(G,"genes"),
          par.settings=list(layout.heights=list(axis.top=0.5)))
dev.off()
