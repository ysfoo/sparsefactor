// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

#include <RcppArmadillo.h>
#include <Rcpp/Benchmark/Timer.h>
#include <gsl/gsl_math.h>
#include <cmath>
#include <vector>
#include "cavifull.h"
#include "digamma.h"

#define EPS 1e-14

using namespace Rcpp;

void initialise(arma::mat &ymat, arma::umat &vmat, arma::vec &pivec,
                arma::mat &lmeans, arma::mat &lsigs,
                arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
                arma::vec &taushapes, arma::vec &taurates,
                arma::vec &alphashapes, arma::vec &alpharates);
void update_lz(arma::mat &ymat, arma::umat &vmat, arma::vec &pivec,
               arma::mat &lmeans, arma::mat &lsigs,
               arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
               arma::vec &taushapes, arma::vec &taurates,
               arma::vec &alphashapes, arma::vec &alpharates);
void update_f(arma::mat &ymat, arma::umat &vmat, arma::mat &lmeans, arma::mat &lsigs,
              arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
              arma::vec &taushapes, arma::vec &taurates);
void update_tau(arma::mat &ymat, arma::umat &vmat, arma::mat &lmeans, arma::mat &lsigs,
                arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
                arma::vec &taushapes, arma::vec &taurates,
                double ptaushape, double ptaurate);
void update_alpha(arma::mat &lmeans, arma::mat &lsigs, arma::mat &zmeans,
                  arma::vec &alphashapes, arma::vec &alpharates,
                  double palphashape, double palpharate);
void permute(arma::vec &pivec,
             arma::mat &lmeans, arma::mat &lsigs,
             arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
             arma::vec &alphashapes, arma::vec &alpharates);
double calc_elbo(arma::mat &ymat, arma::umat &vmat, arma::vec &pivec,
                 arma::mat &lmeans, arma::mat &lsigs,
                 arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
                 arma::vec &taushapes, arma::vec &taurates,
                 arma::vec &alphashapes, arma::vec &alpharates,
                 double ptaushape, double ptaurate,
                 double palphashape, double palpharate);


// [[Rcpp::export]]
// VI entry point, handles NAs
// `check` is how often elbo is calculated and checked for convergence as full elbo calculation is expensive
// `save` is how often the parameters are saved
// algorithm terminates if max difference of z between two iterations is below `tol_z`
// `tol_z` of 0 means the above is effectively not checked for
List cavi(arma::mat ymat, arma::vec &pivec,
          double ptaushape, double ptaurate,
          double palphashape, double palpharate,
          int check=100, int save=0, int max_iter=5000,
          double tol_elbo=1e-14, double tol_z=0, int seed=-1) {
    Timer timer;

    // handle NAs
    arma::uvec na_idx = arma::find_nonfinite(ymat);

    // if no NAs, use cavi_full, which is faster
    if(na_idx.n_elem == 0) return cavi_full(ymat, pivec,
                                             ptaushape, ptaurate, palphashape, palpharate,
                                             check, save, max_iter, tol_elbo, tol_z, seed);
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
    arma::mat fsigs(K, N);
    arma::mat zmeans(G, K);
    arma::vec taushapes(G);
    arma::vec taurates(G);
    arma::vec alphashapes(K);
    arma::vec alpharates(K);

    // M is max length (capacity), m is actual length of output
    int M = 0, m = 0;
    if(save) M = max_iter / save + 1;

    // store in-between parameters
    arma::cube slmeans(M, G, K);
    arma::cube slsigs(M, G, K);
    arma::cube sfmeans(M, K, N);
    arma::cube sfsigs(M, K, N);
    arma::cube szmeans(M, G, K);
    arma::mat staushapes(M, G);
    arma::mat staurates(M, G);
    arma::mat salphashapes(M, K);
    arma::mat salpharates(M, K);
    arma::vec selbo(M);
    arma::vec siter(M);

    // l does not need to be initialised
    initialise(ymat, vmat, pivec, lmeans, lsigs, fmeans, fsigs, zmeans,
               taushapes, taurates, alphashapes, alpharates);

    List params;
    int iter = 0;
    double diff_elbo = 2 * tol_elbo;
    double diff_z = 2 * tol_z;
    double rel_diff = 1;
    double prev_elbo, curr_elbo;
    arma::mat prev_z;
    while((iter < max_iter) && (diff_elbo >= tol_elbo) && (rel_diff >= EPS) && (diff_z >= tol_z)) {
        iter++;
        if(iter % check == 0) {
            prev_z = zmeans;
            prev_elbo = calc_elbo(ymat, vmat, pivec, lmeans, lsigs, fmeans, fsigs, zmeans,
                                  taushapes, taurates, alphashapes, alpharates,
                                  ptaushape, ptaurate, palphashape, palpharate);
        }

        // update parameters
        update_lz(ymat, vmat, pivec, lmeans, lsigs, fmeans, fsigs, zmeans,
                  taushapes, taurates, alphashapes, alpharates);
        update_f(ymat, vmat, lmeans, lsigs, fmeans, fsigs,
                 zmeans, taushapes, taurates);
        update_tau(ymat, vmat, lmeans, lsigs, fmeans, fsigs, zmeans,
                   taushapes, taurates, ptaushape, ptaurate);
        update_alpha(lmeans, lsigs, zmeans, alphashapes, alpharates,
                     palphashape, palpharate);
        // shuffle factors to match sparsity hyperparameters (questionable)
        permute(pivec, lmeans, lsigs, fmeans, fsigs, zmeans, alphashapes, alpharates);

        // for convergence
        if(iter % check == 0) {
            curr_elbo = calc_elbo(ymat, vmat, pivec, lmeans, lsigs, fmeans, fsigs, zmeans,
                                  taushapes, taurates, alphashapes, alpharates,
                                  ptaushape, ptaurate, palphashape, palpharate);
            diff_elbo = fabs(curr_elbo - prev_elbo);
            rel_diff = fabs(diff_elbo / curr_elbo);
            diff_z = arma::abs(zmeans - prev_z).max();
        }

        // save in-between parameters
        if(save && (iter % save == 0)) {
            // calculate elbo if it wasn't calculated this iteration
            if(iter % check) {
                curr_elbo = calc_elbo(ymat, vmat, pivec, lmeans, lsigs, fmeans, fsigs, zmeans,
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
            selbo(m) = curr_elbo;
            siter(m) = iter;
            m++;
            timer.step(std::to_string(m));
        }
    }


    // exited while loop

    // calculate elbo if it wasn't calculated at the final iteration
    if(iter % check) {
        curr_elbo = calc_elbo(ymat, vmat, pivec, lmeans, lsigs, fmeans, fsigs, zmeans,
                              taushapes, taurates, alphashapes, alpharates,
                              ptaushape, ptaurate, palphashape, palpharate);
    }

    if(save) {
        // store the last parameters if it wasn't stored at the final iteration
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
            selbo(m) = curr_elbo;
            siter(m) = iter;
            m++;
            timer.step(std::to_string(m));
        }
        slmeans.resize(m, G, K);
        slsigs.resize(m, G, K);
        sfmeans.resize(m, K, N);
        sfsigs.resize(m, K, N);
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
                              Named("elbo")=curr_elbo,
                              Named("iter")=iter);
    }

    return params;
}

// random initialisation
void initialise(arma::mat &ymat, arma::umat &vmat, arma::vec &pivec,
               arma::mat &lmeans, arma::mat &lsigs,
               arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
               arma::vec &taushapes, arma::vec &taurates,
               arma::vec &alphashapes, arma::vec &alpharates) {
    arma::uword G = ymat.n_rows;
    arma::uword N = ymat.n_cols;
    arma::uword K = pivec.n_elem;

    // initialise f
    fmeans.randn();
    fsigs.ones();

    // initialise z and alphashape
    zmeans.ones();
    for(int k = 0; k < K; k++) {
        alphashapes(k) = 0.5 * G;
    }

    // initialise l, tau and alpha
    arma::mat fz, fzt;
    arma::uvec vi_idx;
    for(arma::uword i = 0; i < G; i++) {
        vi_idx = arma::find(vmat.row(i));
        fz = fmeans.cols(vi_idx);
        fzt = fz.t();
        lmeans.row(i) = ymat(arma::uvec({i}), vi_idx) * fzt * inv_sympd(fz * fzt);
    }
    lsigs.fill(1);

    arma::mat res = ymat - lmeans * fmeans;
    arma::vec tauest = arma::clamp(arma::sum(vmat, 1) - K, 1, N) / sum(res % res % vmat, 1);
    arma::mat alphaest = 1 / var(lmeans, 0, 0).t();

    taushapes.fill(0.5 * N);
    taurates = taushapes / tauest;
    alpharates = alphashapes / alphaest;
}

void update_lz(arma::mat &ymat, arma::umat &vmat, arma::vec &pivec,
               arma::mat &lmeans, arma::mat &lsigs,
               arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
               arma::vec &taushapes, arma::vec &taurates,
               arma::vec &alphashapes, arma::vec &alpharates) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword N = ymat.n_cols;
    arma::uword K = lmeans.n_cols;

    arma::mat poff = arma::repmat(arma::log(1 - pivec.t()), G, 1);
    arma::vec pon, pmax;

    arma::vec tau_bar = taushapes / taurates;
    arma::rowvec alpha_bar = (alphashapes / alpharates).t();

    arma::mat ff_bar = fmeans * fmeans.t();
    arma::vec ff_vec = arma::sum(arma::square(fmeans) + fsigs, 1);
    arma::cube ff_bars(K, K, G);

    arma::uvec vi_idx;
    arma::mat ff_na(K, K);
    arma::vec fsig_na(K);

    for(int i = 0; i < G; i++) {
        vi_idx = arma::find(vmat.row(i));
        if(vi_idx.n_elem == N) {
            ff_bars.slice(i) = ff_bar;
            lsigs.row(i) = 1 / (tau_bar(i) * ff_vec.t() + alpha_bar);
        } else {
            ff_na.zeros();
            fsig_na.zeros();
            for(int j = 0; j < N; j++) {
                if(!vmat(i, j)) {
                    ff_na += fmeans.col(j) * fmeans.col(j).t();
                    fsig_na += fsigs.col(j);
                }
            }
            ff_bars.slice(i) = ff_bar - ff_na;
            lsigs.row(i) = 1 / (tau_bar(i) * (ff_vec - ff_na.diag() - fsig_na).t() + alpha_bar);
        }
    }

    arma::vec tmpvec(G);
    for(arma::uword k : arma::randperm(K)) {
        tmpvec.zeros();
        for(arma::uword k1 = 0; k1 < K; k1++) {
            if(k1 == k) continue;
            tmpvec += arma::vec(ff_bars.tube(k, k1)) % lmeans.col(k1) % zmeans.col(k1);
        }
        lmeans.col(k) = tau_bar % lsigs.col(k) % (ymat * fmeans.row(k).t() - tmpvec);

        pon = log(pivec(k)) + 0.5 * (digammal(alphashapes(k)) - log(2 * M_PI * alpharates(k))
                                         + arma::square(lmeans.col(k)) / lsigs.col(k)
                                         + arma::log(lsigs.col(k)) + log(2 * M_PI));
                                         pmax = arma::max(poff.col(k), pon);
                                         zmeans.col(k) = arma::exp(pon - pmax - arma::log(arma::exp(poff.col(k) - pmax)
                                                                                              + arma::exp(pon - pmax)));
                                         // debugging
                                         for(int i = 0; i < G; i++) {
                                             if(!(zmeans(i, k) > -1)) {
                                                 Rcerr << i << " " << k << " " << lsigs(i, k) << "\n" << tmpvec << '\n';
                                                 return;
                                             }
                                         }
    }
}

void update_f(arma::mat &ymat, arma::umat &vmat, arma::mat &lmeans, arma::mat &lsigs,
                   arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
                   arma::vec &taushapes, arma::vec &taurates) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword N = ymat.n_cols;
    arma::uword K = lmeans.n_cols;

    arma::vec tau_bar = taushapes / taurates;
    arma::mat lzmeans = lmeans % zmeans;
    arma::mat ltaul_bar = lzmeans.t() * (lzmeans.each_col() % tau_bar);
    arma::mat tmpmat = ((arma::square(lmeans) + lsigs) % zmeans).t();
    arma::vec ltaul_vec = tmpmat * tau_bar;
    arma::cube ltaul_bars(K, K, N);

    arma::uvec vj_idx;
    arma::mat ltaul_na(K, K);
    arma::vec lsigtau_na(K);

    for(int j = 0; j < N; j++) {
        vj_idx = arma::find(vmat.col(j));
        if(vj_idx.n_elem == G) {
            ltaul_bars.slice(j) = ltaul_bar;
            fsigs.col(j) = 1 / (ltaul_vec + 1);
        } else {
            ltaul_na.zeros();
            lsigtau_na.zeros();
            for(int i = 0; i < G; i++) {
                if(!vmat(i, j)) {
                    ltaul_na += lzmeans.row(i).t() * lzmeans.row(i) * tau_bar(i);
                    lsigtau_na += tmpmat.col(i) * tau_bar(i);
                }
            }
            ltaul_bars.slice(j) = ltaul_bar - ltaul_na;
            fsigs.col(j) = 1 / (ltaul_vec - lsigtau_na + 1);
        }
    }

    arma::rowvec tmpvec(N);
    for(arma::uword k : arma::randperm(K)) {
        tmpvec.zeros();
        for(arma::uword k1 = 0; k1 < K; k1++) {
            if(k1 == k) continue;
            tmpvec += arma::vec(ltaul_bars.tube(k, k1)).t() % fmeans.row(k1);
        }
        fmeans.row(k) = fsigs.row(k) % ((lzmeans.col(k) % tau_bar).t() * ymat - tmpvec);
    }
}

void update_tau(arma::mat &ymat, arma::umat &vmat, arma::mat &lmeans, arma::mat &lsigs,
                     arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
                     arma::vec &taushapes, arma::vec &taurates,
                     double ptaushape, double ptaurate) {
    // define dimensions
    arma::uword G = ymat.n_rows;
    arma::uword N = ymat.n_cols;
    arma::uword K = lmeans.n_cols;

    taushapes = arma::conv_to<arma::vec>::from(arma::sum(vmat, 1)) * 0.5 + ptaushape;

    arma::vec yy = arma::sum(arma::square(ymat), 1);
    arma::vec lfy_bar = arma::sum(((lmeans % zmeans) * fmeans) % ymat, 1);
    arma::mat ff_bar = fmeans * fmeans.t();
    ff_bar.diag() += arma::sum(fsigs, 1);
    arma::cube ff_bars(K, K, G);

    arma::uvec vi_idx;
    arma::mat ff_na(K, K);

    for(int i = 0; i < G; i++) {
        vi_idx = arma::find(vmat.row(i));
        if(vi_idx.n_elem == N) {
            ff_bars.slice(i) = ff_bar;
        } else {
            ff_na.zeros();
            for(int j = 0; j < N; j++) {
                if(!vmat(i, j)) {
                    ff_na += fmeans.col(j) * fmeans.col(j).t();
                    ff_na.diag() += fsigs.col(j);
                }
            }
            ff_bars.slice(i) = ff_bar - ff_na;
        }
    }

    arma::vec lffl_bar = arma::zeros<arma::vec>(G);
    for(int k = 0; k < K; k++) {
        lffl_bar += zmeans.col(k) % (lsigs.col(k) + arma::square(lmeans.col(k))) % arma::vec(ff_bars.tube(k, k));
        for(int k1 = 0; k1 < k; k1++) {
            lffl_bar += 2 * zmeans.col(k) % zmeans.col(k1) % lmeans.col(k) % lmeans.col(k1) % arma::vec(ff_bars.tube(k, k1));
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

void permute(arma::vec &pivec,
                  arma::mat &lmeans, arma::mat &lsigs,
                  arma::mat &fmeans, arma::mat &fsigs, arma::mat &zmeans,
                  arma::vec &alphashapes, arma::vec &alpharates) {
    arma::uword K = pivec.n_elem;
    arma::rowvec piest = arma::sum(zmeans, 0);
    arma::uvec ord = arma::sort_index(pivec);
    arma::uvec estord = arma::sort_index(piest);
    arma::uvec permute(K);
    for(int k = 0; k < K; k++) {
        permute(ord(k)) = estord(k);
    }
    lmeans = lmeans.cols(permute);
    lsigs = lsigs.cols(permute);
    fmeans = fmeans.rows(permute);
    fsigs = fsigs.rows(permute);
    zmeans = zmeans.cols(permute);
    alphashapes = alphashapes(permute);
    alpharates = alpharates(permute);
}

double calc_elbo(arma::mat &ymat, arma::umat &vmat, arma::vec &pivec,
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

    double elbo = -0.5 * log(2 * M_PI) * arma::accu(vmat);
                  + G * (ptaushape * log(ptaurate) - lgamma(ptaushape))
                  + K * (palphashape * log(palpharate) - lgamma(palphashape));

    elbo += arma::accu((0.5 * arma::conv_to<arma::vec>::from(arma::sum(vmat, 1)) + ptaushape)
                       % (psi_tau - arma::log(taurates)));

    arma::vec tau_bar = taushapes / taurates;
    arma::vec yy = arma::sum(arma::square(ymat), 1);
    arma::vec lfy_bar = arma::sum(((lmeans % zmeans) * fmeans) % ymat, 1);
    arma::mat ff_bar = fmeans * fmeans.t();
    ff_bar.diag() += arma::sum(fsigs, 1);
    arma::cube ff_bars(K, K, G);

    arma::uvec vi_idx;
    arma::mat ff_na(K, K);

    for(int i = 0; i < G; i++) {
        vi_idx = arma::find(vmat.row(i));
        if(vi_idx.n_elem == N) {
            ff_bars.slice(i) = ff_bar;
        } else {
            ff_na.zeros();
            for(int j = 0; j < N; j++) {
                if(!vmat(i, j)) {
                    ff_na += fmeans.col(j) * fmeans.col(j).t();
                    ff_na.diag() += fsigs.col(j);
                }
            }
            ff_bars.slice(i) = ff_bar - ff_na;
        }
    }

    arma::vec lffl_bar = arma::zeros<arma::vec>(G);
    for(int k = 0; k < K; k++) {
        lffl_bar += zmeans.col(k) % (lsigs.col(k) + arma::square(lmeans.col(k))) % arma::vec(ff_bars.tube(k, k));
        for(int k1 = 0; k1 < k; k1++) {
            lffl_bar += 2 * zmeans.col(k) % zmeans.col(k1) % lmeans.col(k) % lmeans.col(k1) % arma::vec(ff_bars.tube(k, k1));
        }
    }
    elbo -= 0.5 * arma::accu(tau_bar % (yy - 2 * lfy_bar + lffl_bar));

    elbo += arma::accu((arma::repmat((psi_alpha - arma::log(2 * M_PI * alpharates)).t(), G, 1)
                            - (arma::square(lmeans) + lsigs) % arma::repmat((alphashapes / alpharates).t(), G, 1))
                           % zmeans) / 2;

    double tmp;
    for(int k = 0; k < K; k++) {
        tmp = arma::accu(zmeans.col(k));
        elbo += tmp * log(pivec(k)) + (G - tmp) * log(1 - pivec(k));
    }

    elbo -= 0.5 * (arma::accu(arma::square(fmeans)) + arma::accu(fsigs));

    elbo += palphashape * arma::accu(psi_alpha - arma::log(alpharates));

    elbo -= arma::accu(tau_bar) * ptaurate + arma::accu(alphashapes / alpharates) * palpharate;

    elbo += arma::accu(zmeans % (arma::log(2 * M_PI * lsigs) + 1)) / 2;
    arma::mat tmpmat = zmeans % arma::log(zmeans) + (1 - zmeans) % arma::log(1 - zmeans);
    elbo -= arma::accu(tmpmat.elem(arma::find_finite(tmpmat)));

    elbo += 0.5 * (K * N + arma::accu(arma::log(fsigs)));

    elbo += arma::accu(taushapes + arma::lgamma(taushapes) - taushapes % psi_tau);
    elbo += arma::accu(alphashapes + arma::lgamma(alphashapes) - alphashapes % psi_alpha);

    return elbo;
}

