#include <RcppArmadillo.h>
#include "lap.h"

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

double update_lap(arma::mat &nus, arma::umat &sigmas, int t, arma::cube &fmats,
                  arma::mat &mf, arma::mat &sf, bool print_mat=false);

//' Relabel factors within MCMC samples
//'
//' Takes a list of samples and relabels the factors such that each entry of F resembles a normal distribution.
//'
//' The negative log-likelihood is iteratively minimised by solving linear assignment problems via the Jonker-Volgenant algorithm. The algorithm is implemented by Tomas Kazmar (https://github.com/gatagat/lap).
//'
//' @param samples A list of \eqn{T} MCMC samples as returned by \code{\link{gibbs}}.
//' \describe{
//' \item{lmat}{\eqn{T}-by-\eqn{G}-by-\eqn{K} array of the sampled loading factors.}
//' \item{fmat}{\eqn{T}-by-\eqn{K}-by-\eqn{N} array of the sampled activation weights.}
//' \item{zmat}{\eqn{T}-by-\eqn{G}-by-\eqn{K} array of the sampled connectivity structures.}
//' \item{tau}{\eqn{T}-by-\eqn{G} matrix of the sampled feature-specific precisions of the noise.}
//' \item{alpha}{\eqn{T}-by-\eqn{K} matrix of the sampled factor-specific precisions of the loading factors.}
//' \item{time}{Vector of sampling times of when samples were generated.}
//' }
//' @param tol Relabelling algorithm terminates when the change in likelihood is smaller than \code{tol}.
//' @param print_action Boolean for whether to print the final means and variances of the target normal distributions.
//' @param print_cost Boolean for whether to print the negative log-likelihood at each iteration.
//' @param to_clone Boolean for whether to copy \code{samples}.
//'
//' @return A modified copy of \code{samples} with relabelled factors.
//'
//' @export
// [[Rcpp::export]]
List relabel_samples(List samples, double tol=1e-8,
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

//' Relabel factors of posterior summary to match a target
//'
//' Takes posterior mean and variance of the activation matrix and finds the relabelling needed for it to match a target (e.g. simualted dataset).
//'
//' The negative log-likelihood is minimised by solving a linear assignment problem via the Jonker-Volgenant algorithm. The algorithm is implemented by Tomas Kazmar (https://github.com/gatagat/lap).
//'
//' @param fmeans Posterior mean of the activation matrix.
//' @param fsigs Posterior variance of the activation matrix.
//' @param fmat Target activation matrix.
//' @param print_mat Boolean for whether to print the cost matrix of the underlying linear assignment problem.
//'
//' @return A list of the permutation and signflips needed for the posterior summary to match the target. The permutation should be applied before the signflips.
//' \item{permutation}{Permutation to apply to the factors of the posterior summary.}
//' \item{sign}{Vector of 1s and -1s to multiply to the factors of the posterior summary after applying the permutation.}
//'
//' @export
// [[Rcpp::export]]
List relabel_params(arma::mat &fmeans, arma::mat &fsigs, arma::mat &fmat,
                    bool print_mat=false) {
    params = clone(params);
    arma::mat lmat = params["lmat"];
    arma::mat fmat = params["fmat"];
    arma::umat zmat = params["zmat"];
    arma::vec tauvec = params["tauvec"];
    arma::vec alphavec = params["alphavec"];

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

    // cost_t BAD = LARGE / (K + 1);

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
        sig(x[k]) = k;
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

    return List::create(Named("permutation")=sig,
                        Named("sign")=nu)
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

    // cost_t BAD = LARGE / (K + 1);

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
