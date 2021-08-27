// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

#include <RcppArmadillo.h>
#include <Rcpp/Benchmark/Timer.h>
#include <gsl/gsl_math.h>
#include <cmath>

using namespace Rcpp;

void initialise_full(arma::mat &ymat, arma::vec &pivec, arma::mat &lmat,
                arma::umat &zmat, arma::mat &fmat,
                arma::vec &tauvec, arma::vec &alphavec);

void sample_z_full(arma::mat &ymat, arma::vec &pivec, arma::umat &zmat,
              arma::mat &fmat, arma::vec &tauvec, arma::vec &alphavec);
void sample_l_full(arma::mat &ymat, arma::mat &lmat, arma::umat &zmat,
              arma::mat &fmat, arma::vec &tauvec, arma::vec &alphavec);
void sample_f_full(arma::mat &ymat, arma::mat &lmat, arma::mat &fmat,
              arma::vec &tauvec);
void sample_tau_full(arma::mat &ymat, arma::mat &lmat, arma::mat &fmat,
                arma::vec &tauvec, double ptaushape, double ptaurate);
void sample_alpha_full(arma::mat &lmat, arma::umat &zmat, arma::vec &alphavec,
                  double palphashape, double palpharate);

double calc_pz_full(arma::uword i, arma::mat &ymat, arma::umat &zmat, arma::mat &fmat,
               double tau, arma::vec &alphavec);

//' MCMC for the sparse factor model with no missing data
//'
//' Runs a MCMC using a collapsed Gibbs sampler, where \strong{L} is marginalised out of the conditional distribution of \strong{Z}.
//'
//' Same functionality as \code{\link{gibbs}}, except that \code{ymat} must have no \code{NA}s.
List gibbs_full(int n_samples, arma::mat &ymat, arma::vec &pivec,
           double ptaushape, double ptaurate,
           double palphashape, double palpharate,
           int burn_in=0, int thin=1,
           int seed=-1) {
    Timer timer;

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
    initialise_full(ymat, pivec, lmat, zmat, fmat, tauvec, alphavec);

    // sample
    int ii = 0;
    for(arma::uword i = 0; i < N_TOT; i++) {
        sample_z_full(ymat, pivec, zmat, fmat, tauvec, alphavec);
        sample_l_full(ymat, lmat, zmat, fmat, tauvec, alphavec);
        sample_f_full(ymat, lmat, fmat, tauvec);
        sample_tau_full(ymat, lmat, fmat, tauvec, ptaushape, ptaurate);
        sample_alpha_full(lmat, zmat, alphavec, palphashape, palpharate);

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

void initialise_full(arma::mat &ymat, arma::vec &pivec,
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
    arma::uvec zi_idx;
    arma::mat fz, fzt;
    lmat.zeros();
    for(arma::uword i = 0; i < G; i++) {
        zi_idx = arma::find(zmat.row(i));
        if(zi_idx.n_elem == 0) continue;
        fz = fmat.rows(zi_idx);
        fzt = fz.t();

        lmat(arma::uvec({i}), zi_idx) = ymat.row(i) * fzt * inv_sympd(fz * fzt);
    }

    // initialise alpha
    for(int k = 0; k < K; k++) {
        int n = arma::accu(zmat.col(k));
        if(n <= 1) {
            alphavec(k) = 1;
            continue;
        }
        alphavec(k) = (n - 1) / arma::accu(arma::square(lmat.col(k) - arma::accu(lmat.col(k)) / n));
    }

    // initialise tau
    arma::mat res = ymat - lmat * fmat;
    tauvec = (N - arma::sum(zmat, 1)) / sum(res % res, 1);
    // unused
    // arma::mat fmat_t = fmat.t();
    // lmat = ymat * fmat_t * arma::inv_sympd(fmat * fmat_t);
    // lmat = lmat % zmat;
    // arma::mat tmp = 1 / var(lmat, 0, 0);
    // alphavec = arma::conv_to<arma::vec>::from(tmp);
}

void sample_z_full(arma::mat &ymat, arma::vec &pivec,
              arma::umat &zmat, arma::mat &fmat, arma::vec &tauvec, arma::vec &alphavec) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword K = pivec.n_elem;

    for(arma::uword i = 0; i < G; i++) {
        for(arma::uword k : arma::randperm(K)) {
            zmat(i, k) = 0;
            double pz0 = calc_pz_full(i, ymat, zmat, fmat, tauvec(i), alphavec) + log(1 - pivec(k));
            zmat(i, k) = 1;
            double pz1 = calc_pz_full(i, ymat, zmat, fmat, tauvec(i), alphavec) + log(pivec(k)) + 0.5 * log(alphavec(k) / 2 / M_SQRTPI);
            double pmax = std::max(pz0, pz1);
            zmat(i, k) = arma::randu() < exp(pz1 - pmax - log(exp(pz0 - pmax) + exp(pz1 - pmax)));
        }
    }
}

double calc_pz_full(arma::uword i, arma::mat &ymat, arma::umat &zmat, arma::mat &fmat, double tau, arma::vec &alphavec) {
    arma::uvec zi_idx = arma::find(zmat.row(i));
    if(zi_idx.n_elem == 0) return 0;

    arma::mat f_zi = fmat.rows(zi_idx);
    arma::mat inv_lcov = tau * f_zi * f_zi.t() + arma::diagmat(alphavec.elem(zi_idx));
    arma::vec lmean = tau * arma::inv_sympd(inv_lcov) * f_zi * ymat.row(i).t();

    return (arma::accu((lmean * lmean.t()) % inv_lcov) - arma::log_det(inv_lcov).real()) / 2;
}

void sample_l_full(arma::mat &ymat, arma::mat &lmat, arma::umat &zmat, arma::mat &fmat, arma::vec &tauvec, arma::vec &alphavec) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword K = lmat.n_cols;

    arma::uvec zi_idx;
    arma::mat f_zi, lcov;
    arma::vec lmean;

    for(arma::uword i = 0; i < G; i++) {
        lmat.row(i).zeros();
        arma::uvec zi_idx = arma::find(zmat.row(i));
        if(zi_idx.n_elem == 0) continue;

        f_zi = fmat.rows(zi_idx);
        lcov = arma::inv_sympd(tauvec(i) * f_zi * f_zi.t() + arma::diagmat(alphavec.elem(zi_idx)));
        /*int K1 = lcov.n_rows;
        for(int k = 0; k < K1; k++) {
            for(int kk = 0; kk < k; kk++) {
                lcov(k, kk) = lcov(kk, k);
            }
        }*/
        lmean = tauvec(i) * lcov * f_zi * ymat.row(i).t();
        lmat(arma::uvec({i}), zi_idx) = arma::mvnrnd(lmean, lcov).t();
    }
}

void sample_f_full(arma::mat &ymat, arma::mat &lmat, arma::mat &fmat, arma::vec &tauvec) {
    // define dimensions
    arma::uword N = ymat.n_cols;
    arma::uword K = lmat.n_cols;

    arma::mat lt_tau = (lmat.each_col() % tauvec).t();
    arma::mat fcov = arma::inv_sympd(lt_tau * lmat + arma::eye(K, K));
    /*for(int k = 0; k < K; k++) {
        for(int kk = 0; kk < k; kk++) {
            fcov(k, kk) = fcov(kk, k);
        }
    }*/
    fmat = arma::chol(fcov, "lower") * arma::randn(K, N) + fcov * lt_tau * ymat;
}

void sample_tau_full(arma::mat &ymat, arma::mat &lmat, arma::mat &fmat, arma::vec &tauvec,
                double ptaushape, double ptaurate) {
    // define dimensions
    arma::uword N = ymat.n_cols;
    arma::uword G = lmat.n_rows;

    arma::mat res = arma::square(ymat - lmat * fmat);

    for(arma::uword i = 0; i < G; i++) {
        tauvec(i) = arma::randg(arma::distr_param(ptaushape + 0.5 * N,
                                1.0 / (ptaurate + 0.5 * arma::accu(res.row(i)))));
    }
}

void sample_alpha_full(arma::mat &lmat, arma::umat &zmat, arma::vec &alphavec,
                  double palphashape, double palpharate) {
    // define dimensions
    arma::uword K = lmat.n_cols;

    for(arma::uword k = 0; k < K; k++) {
        arma::vec lvec = lmat.col(k);
        alphavec(k) = arma::randg(arma::distr_param(palphashape + 0.5 * accu(zmat.col(k)),
                                  1.0 / (palpharate + 0.5 * dot(lvec, lvec))));
    }
}
