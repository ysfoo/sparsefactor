burn.thin <- function(samples, burn=0, thin=1) {
    idx <- seq(burn + 1, dim(samples$lmat)[1], thin)
    return ( list(lmat=samples$lmat[idx,,],
                  fmat=samples$fmat[idx,,],
                  zmat=samples$zmat[idx,,],
                  tau=samples$tau[idx,],
                  alpha=samples$alpha[idx,]))
}

