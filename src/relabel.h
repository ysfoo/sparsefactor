#ifndef RELABEL_H
#define RELABEL_H

#include <RcppArmadillo.h>
using namespace Rcpp;

double update_lap(arma::mat &nus, arma::umat &sigmas,
                  int t, bool use_l, bool sign_switch,
                  arma::cube &lmats, arma::cube &fmats, arma::ucube &zmats,
                  arma::mat &ml, arma::mat &sl, arma::mat &mf, arma::mat &sf, arma::mat &pz,
                  bool print_mat=false);
double update_nolap(arma::mat &nus, int t, bool use_l,
                    arma::cube &lmats, arma::cube &fmats, arma::ucube &zmats,
                    arma::mat &ml, arma::mat &sl, arma::mat &mf, arma::mat &sf);

#endif
