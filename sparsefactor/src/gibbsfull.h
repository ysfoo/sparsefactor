#ifndef CAVIFULL_H
#define CAVIFULL_H

#include <RcppArmadillo.h>
using namespace Rcpp;

List gibbs_full(int n_samples, arma::mat &ymat, arma::vec &pivec,
           double ptaushape, double ptaurate,
           double palphashape, double palpharate,
           int burn_in, int thin, int seed);

#endif
