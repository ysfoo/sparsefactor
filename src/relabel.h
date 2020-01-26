#ifndef RELABEL_H
#define RELABEL_H

#include <RcppArmadillo.h>
using namespace Rcpp;

List relabel(List samples, bool sign_switch, bool label_switch);
double update_lap(arma::mat &nus, arma::umat &sigmas, int t, bool sign_switch,
                  arma::cube &lmats, arma::cube &fmats, arma::ucube &zmats,
                  arma::mat &ml, arma::mat &sl, arma::mat &mf, arma::mat &sf, arma::mat &pz);
double update_nolap(arma::mat &nus, int t,
                    arma::cube &lmats, arma::cube &fmats, arma::ucube &zmats,
                    arma::mat &ml, arma::mat &sl, arma::mat &mf, arma::mat &sf);

#endif
