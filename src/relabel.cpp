#include <RcppArmadillo.h>
#include "relabel.h"
#include "lap.h"

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppGSL)]]

// [[Rcpp::export]]
List relabel(List samples, bool sign_switch=true, bool label_switch=true,
             bool p_every=false, bool use_l=true, double tol=1e-6,
             bool print_action=false, bool print_cost=false,
             bool to_clone=true) {
    if(!(sign_switch || label_switch)) return samples;

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
    arma::mat ml(G, K), sl(G, K), mf(K, N), sf(K, N), pz(G, K);
    int n = 0;
    while(fabs(prev_cost - curr_cost) >= tol) {
        // get MLEs
        ml.zeros();
        sl.zeros();
        mf.zeros();
        sf.zeros();
        pz.zeros();
        for(int t = 0; t < T; t++) {
            ml = ml + (arma::mat(lmats.row(t)).each_row() % nus.row(t)).cols(sigmas.row(t));
            mf = mf + (arma::mat(fmats.row(t)).each_col() % nus.row(t).t()).rows(sigmas.row(t));
            pz = pz + arma::umat(zmats.row(t)).cols(sigmas.row(t));
        }
        ml = ml / pz;
        ml.replace(arma::datum::nan, 0);
        mf /= T;
        for(int t = 0; t < T; t++) {
            sl = sl + arma::square((arma::mat(lmats.row(t)).each_row() % nus.row(t)).cols(sigmas.row(t)) - ml)
                        % arma::umat(zmats.row(t)).cols(sigmas.row(t));
            sf = sf + arma::square((arma::mat(fmats.row(t)).each_col() % nus.row(t).t()).rows(sigmas.row(t)) - mf);
        }
        sl = sl / pz;
        sl.replace(arma::datum::nan, 0);
        sf /= T;
        pz /= T;

        if(!p_every) {
            for(int k = 0; k < K; k++) {
                pz.col(k).fill(arma::mean(pz.col(k)));
            }
        }

        // get nu and sigma
        prev_cost = curr_cost;
        curr_cost = 0;
        for(int t = 0; t < T; t++) {
            curr_cost += (label_switch
                          ? update_lap(nus, sigmas, t, use_l, sign_switch,
                                       lmats, fmats, zmats,
                                       ml, sl, mf, sf, pz)
                          : update_nolap(nus, t, use_l,
                                         lmats, fmats, zmats,
                                         ml, sl, mf, sf));
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
        Rcout << ml << '\n';
        Rcout << sl << '\n';
        Rcout << mf << '\n';
        Rcout << sf << '\n';
        Rcout << pz << '\n';
    }

    return List::create(Named("lmat")=lmats,
                        Named("fmat")=fmats,
                        Named("zmat")=zmats,
                        Named("tau")=taus,
                        Named("alpha")=alphas,
                        Named("time")=times);
}

// [[Rcpp::export]]
List relabel_truth(List truth, arma::mat &fmeans, arma::mat &fsigs, arma::mat &pz,
                   bool sign_switch=true, bool print_mat=false) {
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

    double cost, pcost, ncost;
    arma::vec pvec, nvec;
    if(print_mat) Rcout << "cost matrix:\n";
    for(int k = 0; k < K; k++) {
        for(int s = 0; s < K; s++) {
            costmat[k][s] = 0;
            cost = pcost = ncost = 0;
            for(int i = 0; i < G; i++) {
                if(pz(i, k) == 0) {
                    if(zmat(i, s) == 1) {
                        costmat[k][s] = BAD;
                        break;
                    }
                } else if(zmat(i, s) == 1) {
                    cost -= log(pz(i, k));
                } else if(pz(i, k) == 1) {
                    costmat[k][s] = BAD;
                    break;
                } else {
                    cost -= log(1 - pz(i, k));
                }
            }
            if(costmat[k][s] >= BAD) continue;
            tmpvec = fmat.row(s) / tnorms(s) * snorms(k);
            pcost += arma::accu(arma::square(tmpvec - fmeans.row(k)) / fsigs.row(k));
            ncost += arma::accu(arma::square(tmpvec + fmeans.row(k)) / fsigs.row(k));
            if(sign_switch && (ncost < pcost)) {
                tmpnus(k, s) = -1;
                costmat[k][s] = ncost + arma::accu(arma::log(fsigs.row(k))) + cost;
            } else {
                tmpnus(k, s) = 1;
                costmat[k][s] = pcost + arma::accu(arma::log(fsigs.row(k))) + cost;
            }
        }
        if(print_mat) {
            for(int s = 0; s < K; s++) {
                Rcout << costmat[k][s] << ' ';
            }
            Rcout << '\n';
        }
    }

    cost = 0;
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

double update_lap(arma::mat &nus, arma::umat &sigmas, int t, bool use_l, bool sign_switch,
                  arma::cube &lmats, arma::cube &fmats, arma::ucube &zmats,
                  arma::mat &ml, arma::mat &sl, arma::mat &mf, arma::mat &sf, arma::mat &pz,
                  bool print_mat) {
    arma::mat lmat = lmats.row(t);
    arma::mat fmat = fmats.row(t);
    arma::umat zmat = zmats.row(t);

    const uint_t K = nus.n_cols;
    int G = lmat.n_rows;
    cost_t** costmat = new cost_t*[K];
    arma::mat tmpnus(K, K);
    for(int k = 0; k < K; k++) costmat[k] = new cost_t[K];
    int* x = new int[K];
    int* y = new int[K];

    cost_t BAD = LARGE / (K + 1);

    // calculate cost matrix
    double cost, pcost, ncost;
    arma::vec pvec, nvec;
    if(print_mat) Rcout << "cost matrix " << t << ":\n";
    for(int k = 0; k < K; k++) {
        for(int s = 0; s < K; s++) {
            costmat[k][s] = 0;
            cost = pcost = ncost = 0;
            if(use_l) {
                pvec = arma::square(lmat.col(s) - ml.col(k)) / sl.col(k) + arma::log(sl.col(k));
                nvec = arma::square(lmat.col(s) + ml.col(k)) / sl.col(k) + arma::log(sl.col(k));
            }
            for(int i = 0; i < G; i++) {
                if(pz(i, k) == 0) {
                    if(zmat(i, s) == 1) {
                        costmat[k][s] = BAD;
                        break;
                    }
                } else if(zmat(i, s) == 1) {
                    if(use_l) {
                        // ignoring sl = 0
                        if(pvec(i) < arma::datum::inf) {
                            pcost += pvec(i);
                            ncost += nvec(i);
                        }
                    }
                    cost -= log(pz(i, k));
                } else if(pz(i, k) == 1) {
                    costmat[k][s] = BAD;
                    break;
                } else {
                    cost -= log(1 - pz(i, k));
                }
            }
            if(costmat[k][s] >= BAD) continue;
            pcost += arma::accu(arma::square(fmat.row(s) - mf.row(k)) / sf.row(k));
            ncost += arma::accu(arma::square(fmat.row(s) + mf.row(k)) / sf.row(k));
            if(sign_switch && (ncost < pcost)) {
                tmpnus(k, s) = -1;
                costmat[k][s] = ncost + arma::accu(arma::log(sf.row(k))) + cost;
            } else {
                tmpnus(k, s) = 1;
                costmat[k][s] = pcost + arma::accu(arma::log(sf.row(k))) + cost;
            }
        }
        if(print_mat) {
            for(int s = 0; s < K; s++) {
                Rcout << costmat[k][s] << ' ';
            }
            Rcout << '\n';
        }
    }

    cost = 0;
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

double update_nolap(arma::mat &nus, int t, bool use_l,
                    arma::cube &lmats, arma::cube &fmats, arma::ucube &zmats,
                    arma::mat &ml, arma::mat &sl, arma::mat &mf, arma::mat &sf) {
    arma::mat lmat = lmats.row(t);
    arma::mat fmat = fmats.row(t);
    arma::umat zmat = zmats.row(t);
    arma::mat tmpmat;

    arma::uword K = lmat.n_cols;
    arma::vec pcost, ncost;

    if(use_l) {
        tmpmat = arma::square(lmat - ml) % zmat / sl;
        pcost = (arma::sum(tmpmat.replace(arma::datum::nan, 0), 0).t()
                               + arma::sum(arma::square(fmat - mf) / sf, 1));
        tmpmat = arma::square(lmat + ml) % zmat / sl;
        ncost = (arma::sum(tmpmat.replace(arma::datum::nan, 0), 0).t()
                               + arma::sum(arma::square(fmat + mf) / sf, 1));
    } else {
        pcost = arma::sum(arma::square(fmat - mf) / sf, 1);
        ncost = arma::sum(arma::square(fmat + mf) / sf, 1);
    }
    double cost = 0;
    for(int k = 0; k < K; k++) {
        if(pcost(k) < ncost(k)) {
            nus(t, k) = 1;
            cost += pcost(k);
        } else {
            nus(t, k) = -1;
            cost += ncost(k);
        }
    }

    if(use_l) {
        cost += (arma::accu(arma::log(sl.replace(0, 1)) % zmat)
                     + arma::accu(arma::log(sf)));
    } else cost += arma::accu(arma::log(sf));
    return cost;
}
