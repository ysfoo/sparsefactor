#include <RcppArmadillo.h>
#include "relabel.h"

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

// [[Rcpp::export]]
List relabel(List samples, bool sign_switch, bool label_switch, bool to_clone=true) {
    if(!(sign_switch || label_switch)) return samples;

    if(to_clone) samples = clone(samples);

    arma::cube lmats = samples["lmat"];
    arma::cube fmats = samples["fmat"];
    arma::ucube zmats = samples["zmat"];
    arma::mat taus = samples["tau"];
    arma::mat alphas = samples["alpha"];

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
            nus(t, k) = floor(arma::randu() * 2) * 2 - 1;
            // nus(t, k) = 1;
            sigmas.row(t) = arma::randperm(K).t();
            // sigmas(t, k) = k;
        }
    }

    arma::rowvec tmpvec;
    arma::mat tmpmat;
    arma::umat tmpumat;
    for(int t = 0; t < T; t++) {
        // shuffle L
        tmpmat = lmats.row(t);
        lmats.row(t) = (tmpmat.each_row() % nus.row(t)).cols(sigmas.row(t));

        // shuffle f
        tmpmat = fmats.row(t);
        fmats.row(t) = (tmpmat.each_col() % nus.row(t).t()).rows(sigmas.row(t));

        // shuffle z
        tmpumat = zmats.row(t);
        zmats.row(t) = tmpumat.cols(sigmas.row(t));

        // shuffle alpha
        tmpvec = alphas.row(t);
        alphas.row(t) = tmpvec.elem(sigmas.row(t)).t();
    }

    return List::create(Named("lmat")=lmats,
                        Named("fmat")=fmats,
                        Named("zmat")=zmats,
                        Named("tau")=taus,
                        Named("alpha")=alphas);
}
