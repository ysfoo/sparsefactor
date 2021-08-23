

simulate.data <- function(lmat=NULL, tauvec=NULL, fmat=NULL, zmat=NULL, alphavec=NULL,
                          G=20, N=10, K=4, pivec=NULL, snr=NULL,
                          taushape=100, taurate=1, alphashape=1, alpharate=1) {
    # infer dimensions
    if(!is.null(lmat)) {
        ldim <- dim(lmat)
        G <- ldim[1]
        K <- ldim[2]
    }
    if(!is.null(tauvec)) G <- length(tauvec)
    if(!is.null(fmat)) {
        fdim <- dim(fmat)
        K <- fdim[1]
        N <- fdim[2]
    }
    if(!is.null(zmat)) {
        zdim <- dim(zmat)
        G <- zdim[1]
        K <- zdim[2]
    }
    if(!is.null(alphavec)) K <- length(alphavec)
    if(!is.null(pivec)) K <- length(pivec)

    # simulate parameters
    if(is.null(lmat)) {
        if(is.null(zmat)) {
            # first half of factors are dense, second half are sparse
            if(is.null(pivec)) pivec <- c(rep(0.9, K %/% 2), rep(0.1, K - K %/% 2))
            zmat <- sapply(pivec, simulate.z.col, G=G)
        }
        if(is.null(alphavec)) alphavec <- rgamma(K, alphashape, alpharate)
        lmat <- zmat * sapply(alphavec, function(alpha) rnorm(G, 0, 1 / sqrt(alpha)))
    }
    if(is.null(fmat)) fmat <- matrix(rnorm(K * N), nrow=K, ncol=N)

    lf <- lmat %*% fmat

    # simulate tau
    if(is.null(tauvec)) {
        if(is.null(snr)) tauvec <- rgamma(G, taushape, taurate)
        else {
            tauvec <- snr / matrixStats::rowVars(lf)
            tau.na <- !is.finite(tauvec)
            tauvec[tau.na] <- rgamma(sum(tau.na), taushape, taurate)
        }
    }

    # simulate y
    ymat <- lf + t(sapply(tauvec, function(tau) rnorm(N, 0, 1 / sqrt(tau))))

    return (list(ymat=ymat, lmat=lmat, fmat=fmat, zmat=zmat, tauvec=tauvec, alphavec=alphavec))
}

simulate.z.col <- function(p, G) {
    z.col <- rbinom(G, 1, p)
    # ensure column is not all zero
    if(all(z.col == 0)) z.col[runif(1,1,G)] <- 1
    return (z.col)
}
