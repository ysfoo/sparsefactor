#' Confusion matrix for connectivity structure predictions
#'
#' Generates a confusion matrix consisting of proportions of true positives/negatives and false positives/negatives
#'
#' The order of factors must match, relabel first if needed.
#' 
#' @param true.z Binary matrix of true connectivity structure used in data simulation.
#' @param zmeans Matrix of reals between 0 and 1 corresponding to the true connectivity strucutre. Predictions are taken by rounding each entry to the nearest integer.
#'
#' @return A 2-by-2 matrix of proportions of [[true positive, false negatives], [false positives, true negatives]].
#'
#' @export

get.conf.mat <- function(true.z, zmeans) {
    conf.mat <- matrix(nrow=2, ncol=2)
    zpred <- round(zmeans)
    for(i in 0:1) {
        for(j in 0:1) {
            conf.mat[i+1,j+1] <- sum((true.z == 1-i) & (zpred == 1-j))
        }
    }
    return (conf.mat / length(true.z))
}

#' Accuracy of connectivity structure predictions
#'
#' Proportion of correctly predicted entries of the connectivity structure
#'
#' The order of factors must match, relabel first if needed.
#' 
#' @param true.z Binary matrix of true connectivity structure used in data simulation.
#' @param zmeans Matrix of reals between 0 and 1 corresponding to the true connectivity strucutre. Predictions are taken by rounding each entry to the nearest integer.
#'
#' @return The sum of true positives and true negatives (in proportions).
#'
#' @export

get.zacc <- function(true.z, zmeans) {
    conf.mat <- get.conf.mat(true.z, zmeans)
    return (conf.mat[1,1] + conf.mat[2,2])
}


#' Prediction errors of inferred matrices of the sparse factor model
#'
#' Relative root-mean-square error of 
#'
#' The order of factors must match, relabel (including signflips) first if needed. Note that after relabelling, the likelihood is still invariant under rescaling of factors. For a fair comparison, rows of the posterior mean of \strong{F} are scaled to have the same norm as that of the true \strong{F}, and the columns of the posterior mean of \strong{L} are scaled correspondingly (an inverse scaling).
#' 
#' @param true.l Matrix of loading factors used in data simulation.
#' @param true.f Matrix of factor activations used in data simulation.
#' @param lmeans Posterior mean of loading factors. 
#' @param fmeans Posterior mean of factor activations. 
#'
#' @return Vector of three relative root-mean-square errors for \strong{L},\strong{F},\strong{LF}.
#'
#' @export

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

