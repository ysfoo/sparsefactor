#ifndef CAVIFULL_H
#define CAVIFULL_H

#include <RcppArmadillo.h>
using namespace Rcpp;

List cavi_full(arma::mat &ymat, arma::vec &pivec,
               double ptaushape, double ptaurate,
               double palphashape, double palpharate,
               int check, int save, int max_iter,
               double tol_elbo, double tol_z, int seed);

#endif
