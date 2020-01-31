// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

#include <RcppArmadillo.h>
#include <gsl/gsl_math.h>
#include <cmath>
#include "digamma.h"

using namespace Rcpp;

void initialise(arma::mat &ymat, arma::vec &pivec,
                arma::mat &lmeans, arma::mat &lsigs,
                arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
                arma::vec &taushapes, arma::vec &taurates,
                arma::vec &alphashapes, arma::vec &alpharates);
void update_l(arma::mat &ymat, arma::mat &lmeans, arma::mat &lsigs,
              arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
              arma::vec &taushapes, arma::vec &taurates,
              arma::vec &alphashapes, arma::vec &alpharates);
void update_z(arma::vec &pivec, arma::mat &lmeans, arma::mat &lsigs,
              arma::mat &zmeans, arma::vec &alphashapes, arma::vec &alpharates);
void update_f(arma::mat &ymat, arma::mat &lmeans, arma::mat &lsigs,
              arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
              arma::vec &taushapes, arma::vec &taurates);
void update_tau(arma::mat &ymat, arma::mat &lmeans, arma::mat &lsigs,
                arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
                arma::vec &taushapes, arma::vec &taurates,
                double ptaushape, double ptaurate);
void update_alpha(arma::mat &lmeans, arma::mat &lsigs, arma::mat &zmeans,
                  arma::vec &alphashapes, arma::vec &alpharates,
                  double palphashape, double palpharate);

// [[Rcpp::export]]
List cavi(arma::mat &ymat, arma::vec &pivec,
          double ptaushape, double ptaurate,
          double palphashape, double palpharate,
          int max_iter=10, int seed=-1, bool debug=false) {

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

    // initial parameters
    arma::mat lmeans(G, K);
    arma::mat lsigs(G, K);
    arma::mat fmeans(K, N);
    arma::mat fsigs(K, K);
    arma::mat zmeans(G, K);
    arma::vec taushapes(G);
    arma::vec taurates(G);
    arma::vec alphashapes(K);
    arma::vec alpharates(K);

    // l does not need to be initialised
    initialise(ymat, pivec, lmeans, lsigs, fmeans, fsigs, zmeans,
               taushapes, taurates, alphashapes, alpharates);

    int iter = 0;
    while(iter++ < max_iter) {
        update_l(ymat, lmeans, lsigs, fmeans, fsigs, zmeans,
                 taushapes, taurates, alphashapes, alpharates);
        update_z(pivec, lmeans, lsigs, zmeans, alphashapes, alpharates);
        update_f(ymat, lmeans, lsigs, fmeans, fsigs,
                 zmeans, taushapes, taurates);
        update_tau(ymat, lmeans, lsigs, fmeans, fsigs, zmeans,
                   taushapes, taurates, ptaushape, ptaurate);
        update_alpha(lmeans, lsigs, zmeans, alphashapes, alpharates,
                     palphashape, palpharate);
        Rcout << "iter " << iter << "\n";
        Rcout << zmeans;
    }

    List samples = List::create(Named("lmean")=lmeans,
                                Named("lsig")=lsigs,
                                Named("fmean")=fmeans,
                                Named("fsig")=fsigs,
                                Named("zmean")=zmeans,
                                Named("taushape")=taushapes,
                                Named("taurate")=taurates,
                                Named("alphashape")=alphashapes,
                                Named("alpharate")=alpharates);

    return samples;
}

void initialise(arma::mat &ymat, arma::vec &pivec,
                arma::mat &lmeans, arma::mat &lsigs,
                arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
                arma::vec &taushapes, arma::vec &taurates,
                arma::vec &alphashapes, arma::vec &alpharates) {
    arma::uword G = ymat.n_rows;
    arma::uword N = ymat.n_cols;
    arma::uword K = pivec.n_elem;

    // initialise f
    fmeans.randn();
    fsigs.eye();

    // initialise z and alphashape
    for(int k = 0; k < K; k++) {
        zmeans.col(k).fill(pivec(k));
        alphashapes(k) = 0.5 * G * pivec(k);
    }

    // initialise l, tau and alpha
    arma::mat fmat_t = fmeans.t();
    lmeans = ymat * fmat_t * arma::inv_sympd(fmeans * fmat_t);
    lsigs.fill(1);

    arma::mat res = ymat - lmeans * fmeans;
    arma::vec tauest = (N - K) / sum(res % res, 1);
    arma::mat alphaest = 1 / var(lmeans, 0, 0).t();

    taushapes.fill(0.5 * N);
    taurates = taushapes / tauest;
    alpharates = alphashapes / alphaest;
}

void update_l(arma::mat &ymat, arma::mat &lmeans, arma::mat &lsigs,
              arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
              arma::vec &taushapes, arma::vec &taurates,
              arma::vec &alphashapes, arma::vec &alpharates) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword N = ymat.n_cols;
    arma::uword K = lmeans.n_cols;

    arma::vec ff_bar = N * arma::diagvec(fsigs) + arma::sum(arma::square(fmeans), 1);
    arma::vec tau_bar = taushapes / taurates;
    lsigs = 1 / (tau_bar * ff_bar.t()
                 + arma::repmat((alphashapes / alpharates).t(), G, 1));

    arma::mat res = ymat - ((lmeans % zmeans) * fmeans);

    for(arma::uword k : arma::randperm(K)) {
        lmeans.col(k) = ((res + (lmeans.col(k) % zmeans.col(k)) * fmeans.row(k))
                          * fmeans.row(k).t()) % (tau_bar % lsigs.col(k));
    }
}

void update_z(arma::vec &pivec, arma::mat &lmeans, arma::mat &lsigs,
              arma::mat &zmeans, arma::vec &alphashapes, arma::vec &alpharates) {
    // define dimensions
    arma::uword G = lmeans.n_rows;
    arma::uword K = pivec.n_elem;

    arma::mat poff = arma::repmat(1 - pivec.t(), G, 1);
    arma::rowvec tmpvec(K);
    for(int k = 0; k < K; k++) {
        tmpvec(k) = digammal(alphashapes(k)) - log(2 * M_PI * alpharates(k));
    }

    arma::mat tmp = arma::square(lmeans) / lsigs + arma::repmat(tmpvec, G, 1);
    arma::mat pon = arma::repmat(pivec.t(), G, 1) % arma::sqrt(lsigs) % arma::exp(tmp / 2);

    zmeans = pon / (pon + poff);
    zmeans.replace(arma::datum::nan, 1); // because pon entry is inf
}

void update_f(arma::mat &ymat, arma::mat &lmeans, arma::mat &lsigs,
              arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
              arma::vec &taushapes, arma::vec &taurates) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword N = ymat.n_cols;
    arma::uword K = lmeans.n_cols;

    arma::mat ltaul_bar(K, K);
    arma::vec tau_bar = taushapes / taurates;

    for(int k = 0; k < K; k++) {
        ltaul_bar(k, k) = arma::accu(tau_bar % zmeans.col(k)
                                     % (lsigs.col(k) + arma::square(lmeans.col(k))));
        for(int k1 = 0; k1 < k; k1++) {
            ltaul_bar(k, k1) = arma::accu(tau_bar % zmeans.col(k) % zmeans.col(k1)
                                          % lmeans.col(k) % lmeans.col(k1));
            ltaul_bar(k1, k) = ltaul_bar(k, k1);
        }
    }

    if(!ltaul_bar.is_symmetric()) {
        // something went wrong
        Rcout << zmeans << '\n';
    }
    fsigs = arma::inv_sympd(ltaul_bar + arma::eye(K, K));
    fmeans = fsigs * (lmeans % zmeans).t() * (ymat.each_col() % tau_bar);
}

void update_tau(arma::mat &ymat, arma::mat &lmeans, arma::mat &lsigs,
                arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
                arma::vec &taushapes, arma::vec &taurates,
                double ptaushape, double ptaurate) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword N = ymat.n_cols;
    arma::uword K = lmeans.n_cols;

    taushapes.fill(ptaushape + 0.5 * N);

    arma::vec yy = arma::sum(arma::square(ymat), 1);
    arma::vec lfy_bar = arma::sum(((lmeans % zmeans) * fmeans) % ymat, 1);
    arma::mat ff_bar = N * fsigs + fmeans * fmeans.t();
    arma::vec lffl_bar = arma::zeros<arma::vec>(G);
    for(int k = 0; k < K; k++) {
        lffl_bar += zmeans.col(k) % (lsigs.col(k) + arma::square(lmeans.col(k))) * ff_bar(k, k);
        for(int k1 = 0; k1 < k; k1++) {
            lffl_bar += 2 * zmeans.col(k) % zmeans.col(k1) % lmeans.col(k) % lmeans.col(k1) * ff_bar(k, k1);
        }
    }
    taurates = ptaurate + 0.5 * yy - lfy_bar + 0.5 * lffl_bar;
}

void update_alpha(arma::mat &lmeans, arma::mat &lsigs, arma::mat &zmeans,
                  arma::vec &alphashapes, arma::vec &alpharates,
                  double palphashape, double palpharate) {
    alphashapes = palphashape + arma::sum(zmeans, 0).t() / 2;
    alpharates = palpharate + arma::sum(zmeans % (lsigs + arma::square(lmeans)), 0).t() / 2;
}

double calc_elbo() {
    return 0;
}
