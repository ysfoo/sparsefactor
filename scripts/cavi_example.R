library(sparsefactor)


# simulate data according to Section 5.1 of report
S <- 5 # sparse factors
D <- 1 # dense factors
K <- S + D
N <- 100
G <- 800
data <- simulate.data(K=K, N=N, G=G,
                      zmat = matrix(c(rep(0, 30 * G / 40), rep(1, 1 * G / 40), rep(0, 5 * G / 40), rep(1, 1 * G / 40), rep(0, 2 * G / 40),rep(1, 1 * G / 40),
                                      rep(0, 6 * G / 40), rep(1, 1 * G / 40), rep(0, 13 * G / 40), rep(1, 1 * G / 40), rep(0, 1 * G / 40), rep(1, 1 * G / 40), rep(0, 13 * G / 40), rep(1, 3 * G / 40), rep(0, 1 * G / 40),
                                      rep(1, 1 * G / 40), rep(0, 4 * G / 40), rep(1, 1 * G / 40), rep(0, 14 * G / 40), rep(1, 2 * G / 40), rep(0, 8 * G / 40), rep(1, 6 * G / 40), rep(0, 4 * G / 40),
                                      rep(1, 5 * G / 40), rep(0, 15 * G / 40), rep(1, 10 * G / 40), rep(0, 10 * G / 40),
                                      rep(1, 20 * G / 40), rep(0, 20 * G / 40),
                                      rep(1, G)),
                                    nrow=G, ncol=K),
                      alphavec=rep(1, K), snr=5)