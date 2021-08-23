#include <RcppArmadillo.h>
#include "lap.h"

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

double update_lap(arma::mat &nus, arma::umat &sigmas, int t, arma::cube &fmats,
                  arma::mat &mf, arma::mat &sf, bool print_mat=false);

// [[Rcpp::export]]
List relabel(List samples, double tol=1e-8,
             bool print_action=false, bool print_cost=false,
             bool to_clone=true) {
    if(to_clone) samples = clone(samples);

    arma::cube lmats = samples["lmat"];
    arma::cube fmats = samples["fmat"];
    arma::ucube zmats = samples["zmat"];
    arma::mat taus = samples["tau"];
    arma::mat alphas = samples["alpha"];
    arma::vec times = samples["time"];

    // define dimensions
    arma::uword G = lmats.n_cols;
    arma::uword N = fmats.n_slices;
    arma::uword K = alphas.n_cols;
    arma::uword T = alphas.n_rows;

    // initialise signs and permutations to be identity
    arma::mat nus(T, K);
    arma::umat sigmas(T, K);
    for(int t = 0; t < T; t++) {
        for(int k = 0; k < K; k++) {
            // nus(t, k) = floor(arma::randu() * 2) * 2 - 1;
            nus(t, k) = 1;
            // sigmas.row(t) = arma::randperm(K).t();
            sigmas(t, k) = k;
        }
    }

    double prev_cost = 2 * tol, curr_cost = 0;
    arma::mat mf(K, N), sf(K, N);
    int n = 0;
    while(fabs(prev_cost - curr_cost) >= tol) {
        // get MLEs
        mf.zeros();
        sf.zeros();
        for(int t = 0; t < T; t++) {
            mf = mf + (arma::mat(fmats.row(t)).each_col() % nus.row(t).t()).rows(sigmas.row(t));
        }
        mf /= T;
        for(int t = 0; t < T; t++) {
            sf = sf + arma::square((arma::mat(fmats.row(t)).each_col() % nus.row(t).t()).rows(sigmas.row(t)) - mf);
        }
        sf /= T;

        // get nu and sigma
        prev_cost = curr_cost;
        curr_cost = 0;
        for(int t = 0; t < T; t++) {
            curr_cost += update_lap(nus, sigmas, t, fmats, mf, sf);
        }
        if(print_cost) Rcout << "iter " << n++ << ": " << curr_cost << '\n';
    }

    // shuffle final sample
    for(int t = 0; t < T; t++) {
        lmats.row(t) = (arma::mat(lmats.row(t)).each_row() % nus.row(t)).cols(sigmas.row(t));
        fmats.row(t) = (arma::mat(fmats.row(t)).each_col() % nus.row(t).t()).rows(sigmas.row(t));
        zmats.row(t) = arma::umat(zmats.row(t)).cols(sigmas.row(t));
        alphas.row(t) = arma::rowvec(alphas.row(t)).elem(sigmas.row(t)).t();
    }

    if(print_action) {
        Rcout << mf << '\n';
        Rcout << sf << '\n';
    }

    return List::create(Named("lmat")=lmats,
                        Named("fmat")=fmats,
                        Named("zmat")=zmats,
                        Named("tau")=taus,
                        Named("alpha")=alphas,
                        Named("time")=times);
}

// [[Rcpp::export]]
List relabel_truth(List truth, arma::mat &fmeans, arma::mat &fsigs,
                   bool print_mat=false) {
    truth = clone(truth);
    arma::mat lmat = truth["lmat"];
    arma::mat fmat = truth["fmat"];
    arma::umat zmat = truth["zmat"];
    arma::vec tauvec = truth["tauvec"];
    arma::vec alphavec = truth["alphavec"];

    // define dimensions
    arma::uword G = lmat.n_rows;
    arma::uword N = fmat.n_cols;
    arma::uword K = fmat.n_rows;

    arma::uvec sig(K);
    arma::vec nu(K);

    cost_t** costmat = new cost_t*[K];
    arma::mat tmpnus(K, K);
    tmpnus.fill(1);
    for(int k = 0; k < K; k++) costmat[k] = new cost_t[K];
    int* x = new int[K];
    int* y = new int[K];

    cost_t BAD = LARGE / (K + 1);

    arma::rowvec tnorms(K), snorms(K), tmpvec;
    for(int k = 0; k < K; k++) {
        tnorms(k) = arma::norm(fmat.row(k));
        snorms(k) = arma::norm(fmeans.row(k));
    }

    double pcost, ncost;
    arma::vec pvec, nvec;
    if(print_mat) Rcout << "cost matrix:\n";
    for(int k = 0; k < K; k++) {
        for(int s = 0; s < K; s++) {
            if(tnorms(s) && snorms(k)) tmpvec = fmat.row(s) / tnorms(s) * snorms(k);
            else tmpvec = fmat.row(s);

            pcost = arma::accu(arma::square(tmpvec - fmeans.row(k)) / fsigs.row(k));
            ncost = arma::accu(arma::square(tmpvec + fmeans.row(k)) / fsigs.row(k));
            if(ncost < pcost) {
                tmpnus(k, s) = -1;
                costmat[k][s] = ncost + arma::accu(arma::log(fsigs.row(k)));
            } else {
                tmpnus(k, s) = 1;
                costmat[k][s] = pcost + arma::accu(arma::log(fsigs.row(k)));
            }
        }
        if(print_mat) {
            for(int s = 0; s < K; s++) {
                Rcout << costmat[k][s] << ' ';
            }
            Rcout << '\n';
        }
    }

    double cost = 0;
    if(lapjv_internal(K, costmat, x, y)) Rcerr << "error when solving LAP\n";
    if(print_mat) Rcout << "by row: ";
    for(int k = 0; k < K; k++) {
        if(print_mat) Rcout << x[k] << ' ';
        sig(k) = x[k];
        nu(x[k]) = tmpnus(k, x[k]);
        cost += costmat[k][x[k]];
    }
    if(print_mat) Rcout << '\n';

    for(int k = 0; k < K; k++) delete[] costmat[k];
    delete[] costmat;
    delete[] x;
    delete[] y;

    lmat = (lmat.each_row() % nu.t()).cols(sig);
    fmat = (fmat.each_col() % nu).rows(sig);
    zmat = zmat.cols(sig);
    alphavec = alphavec.elem(sig);

    return List::create(Named("lmat")=lmat,
                        Named("fmat")=fmat,
                        Named("zmat")=zmat,
                        Named("tauvec")=tauvec,
                        Named("alphavec")=alphavec);
}

double update_lap(arma::mat &nus, arma::umat &sigmas, int t, arma::cube &fmats,
                  arma::mat &mf, arma::mat &sf, bool print_mat) {
    arma::mat fmat = fmats.row(t);

    const uint_t K = nus.n_cols;
    cost_t** costmat = new cost_t*[K];
    arma::mat tmpnus(K, K);
    for(int k = 0; k < K; k++) costmat[k] = new cost_t[K];
    int* x = new int[K];
    int* y = new int[K];

    cost_t BAD = LARGE / (K + 1);

    // calculate cost matrix
    double pcost, ncost;
    arma::vec pvec, nvec;
    if(print_mat) Rcout << "cost matrix " << t << ":\n";
    for(int k = 0; k < K; k++) {
        for(int s = 0; s < K; s++) {
            costmat[k][s] = 0;
            pcost = arma::accu(arma::square(fmat.row(s) - mf.row(k)) / sf.row(k));
            ncost = arma::accu(arma::square(fmat.row(s) + mf.row(k)) / sf.row(k));
            if(ncost < pcost) {
                tmpnus(k, s) = -1;
                costmat[k][s] = ncost + arma::accu(arma::log(sf.row(k)));
            } else {
                tmpnus(k, s) = 1;
                costmat[k][s] = pcost + arma::accu(arma::log(sf.row(k)));
            }
        }
        if(print_mat) {
            for(int s = 0; s < K; s++) {
                Rcout << costmat[k][s] << ' ';
            }
            Rcout << '\n';
        }
    }

    double cost = 0;
    if(lapjv_internal(K, costmat, x, y)) Rcerr << "error when solving LAP\n";
    if(print_mat) Rcout << "by row: ";
    for(int k = 0; k < K; k++) {
        if(print_mat) Rcout << x[k] << ' ';
        sigmas(t, k) = x[k];
        nus(t, x[k]) = tmpnus(k, x[k]);
        cost += costmat[k][x[k]];
    }
    if(print_mat) Rcout << '\n';

    for(int k = 0; k < K; k++) delete[] costmat[k];
    delete[] costmat;
    delete[] x;
    delete[] y;

    return cost;
}
