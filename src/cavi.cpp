// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

#include <RcppArmadillo.h>
#include <Rcpp/Benchmark/Timer.h>
#include <gsl/gsl_math.h>
#include <cmath>
#include <vector>
#include "digamma.h"

using namespace Rcpp;

void initialise(arma::mat &ymat, arma::vec &pivec,
                arma::mat &lmeans, arma::mat &lsigs,
                arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
                arma::vec &taushapes, arma::vec &taurates,
                arma::vec &alphashapes, arma::vec &alpharates);
void update_lz(arma::mat &ymat, arma::vec &pivec,
               arma::mat &lmeans, arma::mat &lsigs,
               arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
               arma::vec &taushapes, arma::vec &taurates,
               arma::vec &alphashapes, arma::vec &alpharates,
               double ptaushape, double ptaurate,
               double palphashape, double palpharate);
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
double calc_elbo(arma::mat &ymat, arma::vec &pivec,
                 arma::mat &lmeans, arma::mat &lsigs,
                 arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
                 arma::vec &taushapes, arma::vec &taurates,
                 arma::vec &alphashapes, arma::vec &alpharates,
                 double ptaushape, double ptaurate,
                 double palphashape, double palpharate);

// unused
void update_l(arma::mat &ymat, arma::mat &lmeans, arma::mat &lsigs,
              arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
              arma::vec &taushapes, arma::vec &taurates,
              arma::vec &alphashapes, arma::vec &alpharates);
void update_z(arma::vec &pivec, arma::mat &lmeans, arma::mat &lsigs,
              arma::mat &zmeans, arma::vec &alphashapes, arma::vec &alpharates);


// [[Rcpp::export]]
List cavi(arma::mat &ymat, arma::vec &pivec,
          double ptaushape, double ptaurate,
          double palphashape, double palpharate,
          int check=100, int save=0,
          int max_iter=5000, double tol=1e-2,
          int seed=-1, bool debug=false) {
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

    if(N <= K) {
        Rcerr << "Aborted: number of columns of ymat must be larger than size of pivec\n";
        return List::create();
    }
    if(G <= K) {
        Rcerr << "Aborted: number of columns of ymat must be larger than size of pivec\n";
        return List::create();
    }

    // variational parameters
    arma::mat lmeans(G, K);
    arma::mat lsigs(G, K);
    arma::mat fmeans(K, N);
    arma::mat fsigs(K, K);
    arma::mat zmeans(G, K);
    arma::vec taushapes(G);
    arma::vec taurates(G);
    arma::vec alphashapes(K);
    arma::vec alpharates(K);

    int M = 0, m = 0;
    if(save) M = max_iter / save + 1;

    // store in-between parameters
    arma::cube slmeans(M, G, K);
    arma::cube slsigs(M, G, K);
    arma::cube sfmeans(M, K, N);
    arma::cube sfsigs(M, K, K);
    arma::cube szmeans(M, G, K);
    arma::mat staushapes(M, G);
    arma::mat staurates(M, G);
    arma::mat salphashapes(M, K);
    arma::mat salpharates(M, K);
    arma::vec selbo(M);
    arma::vec siter(M);

    // l does not need to be initialised
    initialise(ymat, pivec, lmeans, lsigs, fmeans, fsigs, zmeans,
               taushapes, taurates, alphashapes, alpharates);

    List params;
    int iter = 0;
    double diff = 2 * tol, prev, curr;
    while((iter < max_iter) && (diff >= tol)) {
        iter++;
        if(iter % check == 0) {
            prev = calc_elbo(ymat, pivec, lmeans, lsigs, fmeans, fsigs, zmeans,
                             taushapes, taurates, alphashapes, alpharates,
                             ptaushape, ptaurate, palphashape, palpharate);
        }

        // update parameters
        update_lz(ymat, pivec, lmeans, lsigs, fmeans, fsigs, zmeans,
                  taushapes, taurates, alphashapes, alpharates,
                  ptaushape, ptaurate, palphashape, palpharate);
        update_f(ymat, lmeans, lsigs, fmeans, fsigs,
                 zmeans, taushapes, taurates);
        update_tau(ymat, lmeans, lsigs, fmeans, fsigs, zmeans,
                   taushapes, taurates, ptaushape, ptaurate);
        update_alpha(lmeans, lsigs, zmeans, alphashapes, alpharates,
                     palphashape, palpharate);

        // for elbo convergence
        if(iter % check == 0) {
            curr = calc_elbo(ymat, pivec, lmeans, lsigs, fmeans, fsigs, zmeans,
                             taushapes, taurates, alphashapes, alpharates,
                             ptaushape, ptaurate, palphashape, palpharate);
            diff = curr - prev;
            if(debug) Rcout << "iter " << iter << ": " << curr << '\n';
        }

        // save in-between parameters
        if(save && (iter % save == 0)) {
            if(iter % check) {
                curr = calc_elbo(ymat, pivec, lmeans, lsigs, fmeans, fsigs, zmeans,
                                 taushapes, taurates, alphashapes, alpharates,
                                 ptaushape, ptaurate, palphashape, palpharate);
            }
            slmeans.row(m) = lmeans;
            slsigs.row(m) = lsigs;
            sfmeans.row(m) = fmeans;
            sfsigs.row(m) = fsigs;
            szmeans.row(m) = zmeans;
            staushapes.row(m) = taushapes.t();
            staurates.row(m) = taurates.t();
            salphashapes.row(m) = alphashapes.t();
            salpharates.row(m) = alpharates.t();
            selbo(m) = curr;
            siter(m) = iter;
            m++;
            timer.step(std::to_string(m));
        }
    }


    // final elbo
    if(iter % check) {
        curr = calc_elbo(ymat, pivec, lmeans, lsigs, fmeans, fsigs, zmeans,
                         taushapes, taurates, alphashapes, alpharates,
                         ptaushape, ptaurate, palphashape, palpharate);
    }
    if(debug) Rcout << "final elbo: " << curr << '\n';

    if(save) {
        if(iter % save) {
            slmeans.row(m) = lmeans;
            slsigs.row(m) = lsigs;
            sfmeans.row(m) = fmeans;
            sfsigs.row(m) = fsigs;
            szmeans.row(m) = zmeans;
            staushapes.row(m) = taushapes.t();
            staurates.row(m) = taurates.t();
            salphashapes.row(m) = alphashapes.t();
            salpharates.row(m) = alpharates.t();
            selbo(m) = curr;
            siter(m) = iter;
            m++;
            timer.step(std::to_string(m));
        }
        slmeans.resize(m, G, K);
        slsigs.resize(m, G, K);
        sfmeans.resize(m, K, N);
        sfsigs.resize(m, K, K);
        szmeans.resize(m, G, K);
        staushapes.resize(m, G);
        staurates.resize(m, G);
        salphashapes.resize(m, K);
        salpharates.resize(m, K);
        selbo.resize(m);
        siter.resize(m);
        params = List::create(Named("lmean")=slmeans,
                              Named("lsig")=slsigs,
                              Named("fmean")=sfmeans,
                              Named("fsig")=sfsigs,
                              Named("zmean")=szmeans,
                              Named("taushape")=staushapes,
                              Named("taurate")=staurates,
                              Named("alphashape")=salphashapes,
                              Named("alpharate")=salpharates,
                              Named("elbo")=selbo,
                              Named("iter")=siter,
                              Named("time")=(NumericVector)(timer) / 1e9);
    } else {
        params = List::create(Named("lmean")=lmeans,
                              Named("lsig")=lsigs,
                              Named("fmean")=fmeans,
                              Named("fsig")=fsigs,
                              Named("zmean")=zmeans,
                              Named("taushape")=taushapes,
                              Named("taurate")=taurates,
                              Named("alphashape")=alphashapes,
                              Named("alpharate")=alpharates,
                              Named("elbo")=curr,
                              Named("iter")=iter);
    }

    return params;
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
        zmeans.col(k).fill(0.5);
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

void update_lz(arma::mat &ymat, arma::vec &pivec,
               arma::mat &lmeans, arma::mat &lsigs,
               arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
               arma::vec &taushapes, arma::vec &taurates,
               arma::vec &alphashapes, arma::vec &alpharates,
               double ptaushape, double ptaurate,
               double palphashape, double palpharate) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword N = ymat.n_cols;
    arma::uword K = lmeans.n_cols;

    arma::mat poff = arma::repmat(arma::log(1 - pivec.t()), G, 1);

    arma::vec ff_bar = N * arma::diagvec(fsigs) + arma::sum(arma::square(fmeans), 1);
    arma::vec tau_bar = taushapes / taurates;
    lsigs = 1 / (tau_bar * ff_bar.t()
                 + arma::repmat((alphashapes / alpharates).t(), G, 1));

    arma::vec tmpvec;
    arma::vec pon, pmax;
    double tmp;
    for(arma::uword k : arma::randperm(K)) {
        tmpvec = arma::zeros<arma::vec>(G);
        for(arma::uword k1 = 0; k1 < K; k1++) {
            if(k1 == k) continue;
            tmpvec += (N * fsigs(k, k1) + arma::accu(fmeans.row(k) % fmeans.row(k1))) * (lmeans.col(k1) % zmeans.col(k1));
        }
        lmeans.col(k) = tau_bar % lsigs.col(k) % (ymat * fmeans.row(k).t() - tmpvec);
        tmp = digammal(alphashapes(k)) - log(2 * M_PI * alpharates(k));
        tmpvec = arma::square(lmeans.col(k)) / lsigs.col(k) + tmp;
        pon = log(pivec(k)) + 0.5 * (arma::log(lsigs.col(k)) + tmpvec + log(2 * M_PI));
        pmax = arma::max(poff.col(k), pon);
        zmeans.col(k) = arma::exp(pon - pmax - arma::log(arma::exp(poff.col(k) - pmax)
                                                             + arma::exp(pon - pmax)));
    }
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

double calc_elbo(arma::mat &ymat, arma::vec &pivec,
                 arma::mat &lmeans, arma::mat &lsigs,
                 arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
                 arma::vec &taushapes, arma::vec &taurates,
                 arma::vec &alphashapes, arma::vec &alpharates,
                 double ptaushape, double ptaurate,
                 double palphashape, double palpharate) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword N = ymat.n_cols;
    arma::uword K = pivec.n_elem;

    arma::vec psi_tau(G), psi_alpha(K);
    for(int i = 0; i < G; i++) {
        psi_tau(i) = digammal(taushapes(i));
    }
    for(int k = 0; k < K; k++) {
        psi_alpha(k) = digammal(alphashapes(k));
    }

    double elbo = 0;

    elbo += (0.5 * N + ptaushape - 1) * arma::accu(psi_tau - arma::log(taurates));

    arma::vec tau_bar = taushapes / taurates;
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
    elbo -= 0.5 * arma::accu(tau_bar % (yy - 2 * lfy_bar + lffl_bar));

    elbo += arma::accu((arma::repmat((psi_alpha - arma::log(2 * M_PI * alpharates)).t(), G, 1)
                        - (arma::square(lmeans) + lsigs) % arma::repmat((alphashapes / alpharates).t(), G, 1))
                       % zmeans) / 2;

    double tmp;
    for(int k = 0; k < K; k++) {
        tmp = arma::accu(zmeans.col(k));
        elbo += tmp * log(pivec(k)) + (1 - tmp) * log(1 - pivec(k));
    }

    elbo -= 0.5 * (N * arma::trace(fsigs) + arma::accu(arma::square(fmeans)));

    elbo += (palphashape - 1) * arma::accu(psi_alpha - arma::log(2 * M_PI * alpharates));

    elbo -= arma::accu(tau_bar) * ptaurate + arma::accu(alphashapes / alpharates) * palpharate;

    elbo += arma::accu(zmeans % (arma::log(2 * M_PI * lsigs) + 1)) / 2;
    arma::mat tmpmat = zmeans % arma::log(zmeans) + (1 - zmeans) % arma::log(1 - zmeans);
    elbo -= arma::accu(tmpmat.replace(arma::datum::nan, 0));

    elbo += 0.5 * N * log(arma::det(fsigs));

    elbo += arma::accu(taushapes - arma::log(taurates) + arma::lgamma(taushapes) + (1 - taushapes) % psi_tau);
    elbo += arma::accu(alphashapes - arma::log(alpharates) + arma::lgamma(alphashapes) + (1 - alphashapes) % psi_alpha);

    return elbo;
}

// unused
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

    arma::vec tmpvec;
    for(arma::uword k : arma::randperm(K)) {
        tmpvec = arma::zeros<arma::vec>(G);
        for(arma::uword k1 = 0; k1 < K; k1++) {
            if(k1 == k) continue;
            tmpvec += (N * fsigs(k, k1) + arma::accu(fmeans.row(k) % fmeans.row(k1))) * (lmeans.col(k1) % zmeans.col(k1));
        }
        lmeans.col(k) = tau_bar % lsigs.col(k) % (ymat * fmeans.row(k).t() - tmpvec);
    }
}

void update_z(arma::vec &pivec, arma::mat &lmeans, arma::mat &lsigs,
              arma::mat &zmeans, arma::vec &alphashapes, arma::vec &alpharates) {
    // define dimensions
    arma::uword G = lmeans.n_rows;
    arma::uword K = pivec.n_elem;

    arma::mat poff = arma::repmat(arma::log(1 - pivec.t()), G, 1);
    arma::rowvec tmpvec(K);
    for(int k = 0; k < K; k++) {
        tmpvec(k) = digammal(alphashapes(k)) - log(2 * M_PI * alpharates(k));
    }

    arma::mat tmp = arma::square(lmeans) / lsigs + arma::repmat(tmpvec, G, 1);
    arma::mat pon = arma::repmat(arma::log(pivec.t()), G, 1) + 0.5 * (arma::log(lsigs) + tmp) ;

    arma::mat pmax = arma::max(poff, pon);
    zmeans = arma::exp(pon - pmax - arma::log(arma::exp(poff - pmax) + arma::exp(pon - pmax)));
}
