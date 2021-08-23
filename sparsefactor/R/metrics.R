get.fmeans <- function(samples) {
    return (plyr::aaply(samples$zmat[,,], c(2,3), mean))
}

get.fsigs <- function(samples) {
    return (plyr::aaply(samples$zmat[,,], c(2,3), var))
}

get.zmeans <- function(samples) {
    return (plyr::aaply(samples$zmat[,,], c(2,3), mean))
}

get.pimeans <- function(samples) {
    return (plyr::aaply(samples$zmat[,,], 3, mean))
}

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
            #fmeans[k,] <- fmeans[k,] * sqrt(sum(true.f[k,]^2) / div)
        }
    }
    return (sqrt(sum((fmeans - true.f)^2) / sum(true.f^2)))
}

get.lferr <- function(true.l, true.f, lmeans, fmeans) {
    true.lf <- true.l %*% true.f
    lfmeans <- lmeans %*% fmeans

    for(k in 1:dim(fmeans)[1]) {
        div <- sum(fmeans[k,]^2)
        if(div > 0) {
            scale_factor <- sqrt(sum(true.f[k,]^2) / div)
            fmeans[k,] <- fmeans[k,] * scale_factor
            lmeans[,k] <- lmeans[,k] / scale_factor
        }
    }

    return(c(sqrt(sum((lmeans - true.l)^2) / sum(true.l^2)),
             sqrt(sum((fmeans - true.f)^2) / sum(true.f^2)),
             sqrt(sum((lfmeans - true.lf)^2) / sum(true.lf^2))))
}

