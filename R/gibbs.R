#' @title trace.plot
#' @description
#' Plots trace plot.
#' @name trace.plot
#' @param vec aaa
#' @param vec2 aaa
#' @param start aaa
#'
#' @export
trace.plot <- function(vec, vec2=NULL, start=1) {
    if(!is.null(vec2)) {
        par(mfrow=c(2,1), mar=c(4,2.5,1,1))
    } else par(mfrow=c(1,1), mar=c(4,2.5,1,1))
    len.vec <- length(vec)
    vec <- vec[start:len.vec]
    plot(0, 0, xlab="iteration", type="n",
         xlim=c(start,len.vec), ylim=c(min(vec), max(vec)))
    lines(start:len.vec, vec, type="l")
    if(!is.null(vec2)) {
        vec2 <- vec2[start:len.vec]
        plot(0, 0, xlab="iteration", type="n",
             xlim=c(start,len.vec), ylim=c(min(vec2), max(vec2)))
        lines(start:len.vec, vec2, type="l")
    }
}

burn.thin <- function(samples, burn=0, thin=1) {
    idx <- seq(burn + 1, dim(samples$lmat)[1], thin)
    return ( list(lmat=samples$lmat[idx,,],
                  fmat=samples$fmat[idx,,],
                  zmat=samples$zmat[idx,,],
                  tau=samples$tau[idx,],
                  alpha=samples$alpha[idx,]))
}

