// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

#include <RcppArmadillo.h>
#include <gsl/gsl_math.h>
#include <cmath>
#include "relabel.h"

using namespace Rcpp;

void initialise(arma::mat &ymat, arma::vec &pivec, arma::umat &zmat,
                arma::mat &fmat, arma::vec &tauvec, arma::vec &alphavec);

void sample_z(arma::mat &ymat, arma::vec &pivec, arma::umat &zmat,
              arma::mat &fmat, arma::vec &tauvec, arma::vec &alphavec);
void sample_l(arma::mat &ymat, arma::mat &lmat, arma::umat &zmat,
              arma::mat &fmat, arma::vec &tauvec, arma::vec &alphavec);
void sample_f(arma::mat &ymat, arma::mat &lmat, arma::mat &fmat,
              arma::vec &tauvec);
void sample_tau(arma::mat &ymat, arma::mat &lmat, arma::mat &fmat,
                arma::vec &tauvec, double ptaushape, double ptaurate);
void sample_alpha(arma::mat &lmat, arma::umat &zmat, arma::vec &alphavec,
                  double palphashape, double palpharate);

double calc_pz(arma::uword i, arma::mat &ymat, arma::umat &zmat, arma::mat &fmat,
               double tau, arma::vec &alphavec);

// [[Rcpp::export]]
List gibbs(int n_iter, arma::mat &ymat, arma::vec &pivec,
           double ptaushape, double ptaurate,
           double palphashape, double palpharate,
           bool sign_switch=true, bool label_switch=true,
           int seed=-1) {

    // set random seed
    if(seed != -1) {
        Environment base_env("package:base");
        Function set_seed_r = base_env["set.seed"];
        set_seed_r(seed);
    }

    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword N = ymat.n_cols;
    arma::uword K = pivec.n_elem;

    if(N <= K) {
        Rcerr << "Aborted: number of columns of ymat must be larger than size of pivec\n";
        return List::create();
    }
    if(G <= K) {
        Rcerr << "Aborted: number of columns of ymat must be larger than size of pivec\n";
        return List::create();
    }

    // define parameters
    arma::cube lmats(n_iter, G, K);
    arma::cube fmats(n_iter, K, N);
    arma::ucube zmats(n_iter, G, K);
    arma::mat taus(n_iter, G);
    arma::mat alphas(n_iter, K);

    // initial parameters
    arma::mat lmat(G, K);
    arma::mat fmat(K, N);
    arma::umat zmat(G, K);
    arma::vec tauvec(G);
    arma::vec alphavec(K);

    // l does not need to be initialised
    initialise(ymat, pivec, zmat, fmat, tauvec, alphavec);

    // sample
    for(arma::uword i = 0; i < n_iter; i++) {
        // Rcout << "iteration " << i << endl;
        sample_z(ymat, pivec, zmat, fmat, tauvec, alphavec);
        sample_l(ymat, lmat, zmat, fmat, tauvec, alphavec);
        sample_f(ymat, lmat, fmat, tauvec);
        sample_tau(ymat, lmat, fmat, tauvec, ptaushape, ptaurate);
        sample_alpha(lmat, zmat, alphavec, palphashape, palpharate);

        lmats.row(i) = lmat;
        fmats.row(i) = fmat;
        zmats.row(i) = zmat;
        taus.row(i) = tauvec.t();
        alphas.row(i) = alphavec.t();
    }

    List samples = List::create(Named("lmat")=lmats,
                                Named("fmat")=fmats,
                                Named("zmat")=zmats,
                                Named("tau")=taus,
                                Named("alpha")=alphas);

    return samples;
}

void initialise(arma::mat &ymat, arma::vec &pivec,
                arma::umat &zmat, arma::mat &fmat, arma::vec &tauvec, arma::vec &alphavec) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword N = ymat.n_cols;
    arma::uword K = pivec.n_elem;

    // initialise f
    fmat.randn();

    // initialise z
    arma::mat pimat(K, G);
    for(arma::uword i = 0; i < G; i++) {
        pimat.col(i) = pivec;
    }
    zmat = arma::randu(G, K) < pimat.t();

    // initialise tau and alpha
    arma::mat fmat_t = fmat.t();
    arma::mat ols_lmat = ymat * fmat_t * arma::inv_sympd(fmat * fmat_t);
    arma::mat res = ymat - ols_lmat * fmat;
    tauvec = (N - K) / sum(res % res, 1);
    arma::mat tmp = 1 / var(ols_lmat, 0, 0);
    alphavec = arma::conv_to<arma::vec>::from(tmp);
}

void sample_z(arma::mat &ymat, arma::vec &pivec,
              arma::umat &zmat, arma::mat &fmat, arma::vec &tauvec, arma::vec &alphavec) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword K = pivec.n_elem;

    for(arma::uword i = 0; i < G; i++) {
        for(arma::uword k : arma::randperm(K)) {
            zmat(i, k) = 0;
            double pz0 = calc_pz(i, ymat, zmat, fmat, tauvec(i), alphavec) * (1 - pivec(k));
            zmat(i, k) = 1;
            double pz1 = calc_pz(i, ymat, zmat, fmat, tauvec(i), alphavec) * pivec(k) * sqrt(alphavec(k) / 2) / M_SQRTPI;
            zmat(i, k) = (arma::randu() * (pz0 + pz1)) < pz1;
        }
    }
}

double calc_pz(arma::uword i, arma::mat &ymat, arma::umat &zmat, arma::mat &fmat, double tau, arma::vec &alphavec) {
    arma::uvec zi_idx = find(zmat.row(i));
    if(zi_idx.n_elem == 0) return 1;

    arma::mat f_zi = fmat.rows(zi_idx);
    arma::mat inv_lcov = tau * f_zi * f_zi.t() + arma::diagmat(alphavec.elem(zi_idx));
    arma::vec lmean = tau * arma::inv_sympd(inv_lcov) * f_zi * ymat.row(i).t();

    return exp((accu((lmean * lmean.t()) % inv_lcov) - arma::log_det(inv_lcov).real()) / 2);
}

void sample_l(arma::mat &ymat, arma::mat &lmat, arma::umat &zmat, arma::mat &fmat, arma::vec &tauvec, arma::vec &alphavec) {
    // define dimensions
    arma::uword G = ymat.n_rows;

    for(arma::uword i = 0; i < G; i++) {
        lmat.row(i).zeros();
        arma::uvec zi_idx = find(zmat.row(i));
        if(zi_idx.n_elem == 0) continue;

        arma::mat f_zi = fmat.rows(zi_idx);
        arma::mat lcov = arma::inv_sympd(tauvec(i) * f_zi * f_zi.t() + arma::diagmat(alphavec.elem(zi_idx)));
        arma::vec lmean = tauvec(i) * lcov * f_zi * ymat.row(i).t();
        arma::mat l_zi = mvnrnd(lmean, lcov);

        arma::uword j = 0;
        for(arma::uword idx : zi_idx) {
            lmat(i, idx) = l_zi(j++);
        }
    }
}

void sample_f(arma::mat &ymat, arma::mat &lmat, arma::mat &fmat, arma::vec &tauvec) {
    // define dimensions
    arma::uword N = ymat.n_cols;
    arma::uword K = lmat.n_cols;

    arma::mat lt_tau = (lmat.each_col() % tauvec).t();
    arma::mat fcov = arma::inv_sympd(lt_tau * lmat + arma::eye(K, K));
    fmat = arma::chol(fcov, "lower") * arma::randn(K, N) + fcov * lt_tau * ymat;
}

void sample_tau(arma::mat &ymat, arma::mat &lmat, arma::mat &fmat, arma::vec &tauvec,
                double ptaushape, double ptaurate) {
    // define dimensions
    arma::uword N = ymat.n_cols;
    arma::uword G = lmat.n_rows;

    for(arma::uword i = 0; i < G; i++) {
        arma::rowvec res = ymat.row(i) - lmat.row(i) * fmat;
        tauvec(i) = arma::randg(arma::distr_param(ptaushape + 0.5 * N,
                                1.0 / (ptaurate + 0.5 * dot(res, res))));
    }
}

void sample_alpha(arma::mat &lmat, arma::umat &zmat, arma::vec &alphavec,
                  double palphashape, double palpharate) {
    // define dimensions
    arma::uword K = lmat.n_cols;

    for(arma::uword k = 0; k < K; k++) {
        arma::vec lvec = lmat.col(k);
        alphavec(k) = arma::randg(arma::distr_param(palphashape + 0.5 * accu(zmat.col(k)),
                                  1.0 / (palpharate + 0.5 * dot(lvec, lvec))));
    }
}
