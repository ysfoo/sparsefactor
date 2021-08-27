// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

#include <RcppArmadillo.h>
#include <Rcpp/Benchmark/Timer.h>
#include <gsl/gsl_math.h>
#include <cmath>
#include "gibbsfull.h"

using namespace Rcpp;

void initialise(arma::mat &ymat, arma::umat &vmat, arma::vec &pivec, arma::mat &lmat,
                arma::umat &zmat, arma::mat &fmat,
                arma::vec &tauvec, arma::vec &alphavec);

void sample_z(arma::mat &ymat, arma::umat &vmat, arma::vec &pivec, arma::umat &zmat,
              arma::mat &fmat, arma::vec &tauvec, arma::vec &alphavec);
void sample_l(arma::mat &ymat, arma::umat &vmat, arma::mat &lmat, arma::umat &zmat,
              arma::mat &fmat, arma::vec &tauvec, arma::vec &alphavec);
void sample_f(arma::mat &ymat, arma::umat &vmat, arma::mat &lmat, arma::mat &fmat,
              arma::vec &tauvec);
void sample_tau(arma::mat &ymat, arma::umat &vmat, arma::mat &lmat, arma::mat &fmat,
                arma::vec &tauvec, double ptaushape, double ptaurate);
void sample_alpha(arma::mat &lmat, arma::umat &zmat, arma::vec &alphavec,
                  double palphashape, double palpharate);

double calc_pz(arma::uword i, arma::mat &ymat, arma::umat &vmat, arma::umat &zmat, arma::mat &fmat,
               double tau, arma::vec &alphavec);


//' MCMC for the sparse factor model
//'
//' Runs a MCMC using a collapsed Gibbs sampler, where \strong{L} is marginalised out of the conditional distribution of \strong{Z}.
//'
//' @param n_samples Number of samples in MCMC chain, excluding burn-in samples.
//' @param ymat Data matrix, rows corresponding to features and columns corresponding to samples. May contain \code{NA}s.
//' @param pivec Vector of sparsity hyperparameters for each factor.
//' @param ptaushape Shape hyperparameter of the gamma prior for the feature-specific precision of the noise.
//' @param ptaurate Rate hyperparameter of the gamma prior for the feature-specific precision of the noise.
//' @param palphashape Shape hyperparameter of the gamma prior for the factor-specific precision of the loading factors.
//' @param palpharate Rate hyperparameter of the gamma prior for simulating the factor-specific precision of the loading factors.
//' @param burn_in Number of burn-in samples (these are discarded).
//' @param thin Discard all but one sample for every \code{thin} samples generated.
//' @param seed Random seed. No seed is set when \code{seed} is \code{-1}.
//'
//' @return A list of \code{n_samples} MCMC samples.
//' \describe{
//' \item{lmat}{\eqn{T}-by-\eqn{G}-by-\eqn{K} array of the sampled loading factors.}
//' \item{fmat}{\eqn{T}-by-\eqn{K}-by-\eqn{N} array of the sampled activation weights.}
//' \item{zmat}{\eqn{T}-by-\eqn{G}-by-\eqn{K} array of the sampled connectivity structures.}
//' \item{tau}{\eqn{T}-by-\eqn{G} matrix of the sampled feature-specific precisions of the noise.}
//' \item{alpha}{\eqn{T}-by-\eqn{K} matrix of the sampled factor-specific precisions of the loading factors.}
//' \item{time}{Vector of sampling times (in seconds) of when samples were generated.}
//' }
//'
//' @export
// [[Rcpp::export]]
List gibbs(int n_samples, arma::mat ymat, arma::vec &pivec,
           double ptaushape, double ptaurate,
           double palphashape, double palpharate,
           int burn_in=0, int thin=1,
           int seed=-1) {
    Timer timer;

    // handle NAs
    arma::uvec na_idx = arma::find_nonfinite(ymat);
    if(na_idx.n_elem == 0) return gibbs_full(n_samples, ymat, pivec,
                                             ptaushape, ptaurate, palphashape, palpharate,
                                             burn_in, thin, seed);
    arma::umat vmat = arma::ones<arma::umat>(arma::size(ymat));
    ymat.elem(na_idx).zeros();
    vmat.elem(na_idx).zeros();

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

    int N_TOT = n_samples * thin + burn_in;

    if(N <= K) {
        Rcerr << "Aborted: number of columns of ymat must be larger than size of pivec\n";
        return List::create();
    }
    if(G <= K) {
        Rcerr << "Aborted: number of columns of ymat must be larger than size of pivec\n";
        return List::create();
    }

    // define parameters
    arma::cube lmats(n_samples, G, K);
    arma::cube fmats(n_samples, K, N);
    arma::ucube zmats(n_samples, G, K);
    arma::mat taus(n_samples, G);
    arma::mat alphas(n_samples, K);

    // initial parameters
    arma::mat lmat(G, K);
    arma::mat fmat(K, N);
    arma::umat zmat(G, K);
    arma::vec tauvec(G);
    arma::vec alphavec(K);

    // l does not need to be initialised
    initialise(ymat, vmat, pivec, lmat, zmat, fmat, tauvec, alphavec);

    // sample
    int ii = 0;
    for(arma::uword i = 0; i < N_TOT; i++) {
        sample_z(ymat, vmat, pivec, zmat, fmat, tauvec, alphavec);
        sample_l(ymat, vmat, lmat, zmat, fmat, tauvec, alphavec);
        sample_f(ymat, vmat, lmat, fmat, tauvec);
        sample_tau(ymat, vmat, lmat, fmat, tauvec, ptaushape, ptaurate);
        sample_alpha(lmat, zmat, alphavec, palphashape, palpharate);

        if((i >= burn_in) && ((i - burn_in + 1) % thin == 0)) {
            lmats.row(ii) = lmat;
            fmats.row(ii) = fmat;
            zmats.row(ii) = zmat;
            taus.row(ii) = tauvec.t();
            alphas.row(ii) = alphavec.t();
            ii++;
            timer.step(std::to_string(ii));
        }
    }

    List samples = List::create(Named("lmat")=lmats,
                                Named("fmat")=fmats,
                                Named("zmat")=zmats,
                                Named("tau")=taus,
                                Named("alpha")=alphas,
                                Named("time")=(NumericVector)(timer) / 1e9);

    return samples;
}

void initialise(arma::mat &ymat, arma::umat &vmat, arma::vec &pivec,
                arma::mat &lmat, arma::umat &zmat, arma::mat &fmat,
                arma::vec &tauvec, arma::vec &alphavec) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword N = ymat.n_cols;
    arma::uword K = pivec.n_elem;

    // initialise f
    fmat.randn();

    // initialise z
    zmat = arma::randu(G, K) < arma::repmat(pivec.t(), G, 1);

    // initialise lmat
    arma::mat fz, fzt;
    arma::uvec zi_idx, vi_idx;
    lmat.zeros();
    for(arma::uword i = 0; i < G; i++) {
        vi_idx = arma::find(vmat.row(i));
        zi_idx = arma::find(zmat.row(i));

        if(zi_idx.n_elem == 0) continue;
        fz = fmat(zi_idx, vi_idx);
        fzt = fz.t();
        lmat(arma::uvec({i}), zi_idx) = ymat(arma::uvec({i}), vi_idx) * fzt * inv_sympd(fz * fzt);
    }

    // initialise alpha
    for(int k = 0; k < K; k++) {
        int n = arma::accu(zmat.col(k));
        alphavec(k) = std::max(1, n - 1) / arma::accu(arma::square(lmat.col(k) - arma::accu(lmat.col(k)) / n));
    }

    // initialise tau
    arma::mat res = ymat - lmat * fmat;
    tauvec = arma::clamp(arma::sum(vmat, 1) - arma::sum(zmat, 1), 1, N) / sum(res % res % vmat, 1);

    // unused
    // arma::mat fmat_t = fmat.t();
    // lmat = ymat * fmat_t * arma::inv_sympd(fmat * fmat_t);
    // lmat = lmat % zmat;
    // arma::mat tmp = 1 / var(lmat, 0, 0);
    // alphavec = arma::conv_to<arma::vec>::from(tmp);
}

void sample_z(arma::mat &ymat, arma::umat &vmat, arma::vec &pivec,
              arma::umat &zmat, arma::mat &fmat, arma::vec &tauvec, arma::vec &alphavec) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword K = pivec.n_elem;

    for(arma::uword i = 0; i < G; i++) {
        for(arma::uword k : arma::randperm(K)) {
            zmat(i, k) = 0;
            double pz0 = calc_pz(i, ymat, vmat, zmat, fmat, tauvec(i), alphavec) + log(1 - pivec(k));
            zmat(i, k) = 1;
            double pz1 = calc_pz(i, ymat, vmat, zmat, fmat, tauvec(i), alphavec) + log(pivec(k)) + 0.5 * log(alphavec(k) / 2 / M_SQRTPI);
            double pmax = std::max(pz0, pz1);
            zmat(i, k) = arma::randu() < exp(pz1 - pmax - log(exp(pz0 - pmax) + exp(pz1 - pmax)));
        }
    }
}

double calc_pz(arma::uword i, arma::mat &ymat, arma::umat &vmat, arma::umat &zmat, arma::mat &fmat, double tau, arma::vec &alphavec) {
    arma::uvec vi_idx = arma::find(vmat.row(i));
    arma::uvec zi_idx = arma::find(zmat.row(i));
    if(zi_idx.n_elem == 0) return 0;

    arma::mat f_zi = fmat(zi_idx, vi_idx);
    arma::mat inv_lcov = tau * f_zi * f_zi.t() + arma::diagmat(alphavec.elem(zi_idx));
    arma::vec lmean = tau * arma::inv_sympd(inv_lcov) * f_zi * ymat(arma::uvec({i}), vi_idx).t();

    return (arma::accu((lmean * lmean.t()) % inv_lcov) - arma::log_det(inv_lcov).real()) / 2;
}

void sample_l(arma::mat &ymat, arma::umat &vmat, arma::mat &lmat, arma::umat &zmat, arma::mat &fmat, arma::vec &tauvec, arma::vec &alphavec) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword K = lmat.n_cols;

    arma::uvec zi_idx, vi_idx;
    arma::mat f_zi, lcov;
    arma::vec lmean;

    for(arma::uword i = 0; i < G; i++) {
        lmat.row(i).zeros();

        vi_idx = arma::find(vmat.row(i));
        zi_idx = arma::find(zmat.row(i));
        if(zi_idx.n_elem == 0) continue;

        f_zi = fmat(zi_idx, vi_idx);
        lcov = arma::inv_sympd(tauvec(i) * f_zi * f_zi.t() + arma::diagmat(alphavec.elem(zi_idx)));
        /*int K1 = lcov.n_rows;
        for(int k = 0; k < K1; k++) {
            for(int kk = 0; kk < k; kk++) {
                lcov(k, kk) = lcov(kk, k);
            }
        }*/
        lmean = tauvec(i) * lcov * f_zi * ymat(arma::uvec({i}), vi_idx).t();
        lmat(arma::uvec({i}), zi_idx) = arma::mvnrnd(lmean, lcov).t();
    }
}

void sample_f(arma::mat &ymat, arma::umat &vmat, arma::mat &lmat, arma::mat &fmat, arma::vec &tauvec) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword N = ymat.n_cols;
    arma::uword K = lmat.n_cols;

    arma::uvec vj_idx;
    arma::mat lv, lt_tau, fcov;

    arma::mat lt_tau_full = (lmat.each_col() % tauvec).t();
    arma::mat fcov_full = arma::inv_sympd(lt_tau_full * lmat + arma::eye(K, K));
    arma::mat fchol = arma::chol(fcov_full, "lower");

    for(arma::uword j = 0; j < N; j++) {
        vj_idx = arma::find(vmat.col(j));
        if(vj_idx.n_elem == G) {
            fmat.col(j) = fchol * arma::randn(K) + fcov_full * lt_tau_full * ymat.col(j);
            continue;
        }
        lv = lmat.rows(vj_idx);
        lt_tau = (lv.each_col() % tauvec(vj_idx)).t();
        fcov = arma::inv_sympd(lt_tau * lv + arma::eye(K, K));
        fmat.col(j) = arma::mvnrnd(fcov * lt_tau * ymat(vj_idx, arma::uvec({j})), fcov);
    }
}

void sample_tau(arma::mat &ymat, arma::umat &vmat, arma::mat &lmat, arma::mat &fmat, arma::vec &tauvec,
                double ptaushape, double ptaurate) {
    // define dimensions
    arma::uword N = ymat.n_cols;
    arma::uword G = lmat.n_rows;

    arma::mat res = arma::square(ymat - lmat * fmat) % vmat;

    for(arma::uword i = 0; i < G; i++) {
        tauvec(i) = arma::randg(arma::distr_param(ptaushape + 0.5 * arma::accu(vmat.row(i)),
                                1.0 / (ptaurate + 0.5 * arma::accu(res.row(i)))));
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
