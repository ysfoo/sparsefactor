// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// cavi
List cavi(arma::mat& ymat, arma::vec& pivec, double ptaushape, double ptaurate, double palphashape, double palpharate, int check, int save, int max_iter, double tol, int seed, bool debug);
RcppExport SEXP _sparsefactor_cavi(SEXP ymatSEXP, SEXP pivecSEXP, SEXP ptaushapeSEXP, SEXP ptaurateSEXP, SEXP palphashapeSEXP, SEXP palpharateSEXP, SEXP checkSEXP, SEXP saveSEXP, SEXP max_iterSEXP, SEXP tolSEXP, SEXP seedSEXP, SEXP debugSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type ymat(ymatSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type pivec(pivecSEXP);
    Rcpp::traits::input_parameter< double >::type ptaushape(ptaushapeSEXP);
    Rcpp::traits::input_parameter< double >::type ptaurate(ptaurateSEXP);
    Rcpp::traits::input_parameter< double >::type palphashape(palphashapeSEXP);
    Rcpp::traits::input_parameter< double >::type palpharate(palpharateSEXP);
    Rcpp::traits::input_parameter< int >::type check(checkSEXP);
    Rcpp::traits::input_parameter< int >::type save(saveSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< bool >::type debug(debugSEXP);
    rcpp_result_gen = Rcpp::wrap(cavi(ymat, pivec, ptaushape, ptaurate, palphashape, palpharate, check, save, max_iter, tol, seed, debug));
    return rcpp_result_gen;
END_RCPP
}
// gibbs
List gibbs(int n_samples, arma::mat& data, arma::vec& pivec, double ptaushape, double ptaurate, double palphashape, double palpharate, int burn_in, int thin, int seed);
RcppExport SEXP _sparsefactor_gibbs(SEXP n_samplesSEXP, SEXP dataSEXP, SEXP pivecSEXP, SEXP ptaushapeSEXP, SEXP ptaurateSEXP, SEXP palphashapeSEXP, SEXP palpharateSEXP, SEXP burn_inSEXP, SEXP thinSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n_samples(n_samplesSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type data(dataSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type pivec(pivecSEXP);
    Rcpp::traits::input_parameter< double >::type ptaushape(ptaushapeSEXP);
    Rcpp::traits::input_parameter< double >::type ptaurate(ptaurateSEXP);
    Rcpp::traits::input_parameter< double >::type palphashape(palphashapeSEXP);
    Rcpp::traits::input_parameter< double >::type palpharate(palpharateSEXP);
    Rcpp::traits::input_parameter< int >::type burn_in(burn_inSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(gibbs(n_samples, data, pivec, ptaushape, ptaurate, palphashape, palpharate, burn_in, thin, seed));
    return rcpp_result_gen;
END_RCPP
}
// gibbs_full
List gibbs_full(int n_samples, arma::mat& ymat, arma::vec& pivec, double ptaushape, double ptaurate, double palphashape, double palpharate, int burn_in, int thin, int seed);
RcppExport SEXP _sparsefactor_gibbs_full(SEXP n_samplesSEXP, SEXP ymatSEXP, SEXP pivecSEXP, SEXP ptaushapeSEXP, SEXP ptaurateSEXP, SEXP palphashapeSEXP, SEXP palpharateSEXP, SEXP burn_inSEXP, SEXP thinSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n_samples(n_samplesSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type ymat(ymatSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type pivec(pivecSEXP);
    Rcpp::traits::input_parameter< double >::type ptaushape(ptaushapeSEXP);
    Rcpp::traits::input_parameter< double >::type ptaurate(ptaurateSEXP);
    Rcpp::traits::input_parameter< double >::type palphashape(palphashapeSEXP);
    Rcpp::traits::input_parameter< double >::type palpharate(palpharateSEXP);
    Rcpp::traits::input_parameter< int >::type burn_in(burn_inSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    rcpp_result_gen = Rcpp::wrap(gibbs_full(n_samples, ymat, pivec, ptaushape, ptaurate, palphashape, palpharate, burn_in, thin, seed));
    return rcpp_result_gen;
END_RCPP
}
// relabel
List relabel(List samples, bool sign_switch, bool label_switch, bool p_every, bool use_l, double tol, bool print_action, bool print_cost, bool to_clone);
RcppExport SEXP _sparsefactor_relabel(SEXP samplesSEXP, SEXP sign_switchSEXP, SEXP label_switchSEXP, SEXP p_everySEXP, SEXP use_lSEXP, SEXP tolSEXP, SEXP print_actionSEXP, SEXP print_costSEXP, SEXP to_cloneSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type samples(samplesSEXP);
    Rcpp::traits::input_parameter< bool >::type sign_switch(sign_switchSEXP);
    Rcpp::traits::input_parameter< bool >::type label_switch(label_switchSEXP);
    Rcpp::traits::input_parameter< bool >::type p_every(p_everySEXP);
    Rcpp::traits::input_parameter< bool >::type use_l(use_lSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< bool >::type print_action(print_actionSEXP);
    Rcpp::traits::input_parameter< bool >::type print_cost(print_costSEXP);
    Rcpp::traits::input_parameter< bool >::type to_clone(to_cloneSEXP);
    rcpp_result_gen = Rcpp::wrap(relabel(samples, sign_switch, label_switch, p_every, use_l, tol, print_action, print_cost, to_clone));
    return rcpp_result_gen;
END_RCPP
}
// relabel_truth
List relabel_truth(List truth, arma::mat& fmeans, arma::mat& fsigs, arma::mat& pz, bool sign_switch, bool print_mat);
RcppExport SEXP _sparsefactor_relabel_truth(SEXP truthSEXP, SEXP fmeansSEXP, SEXP fsigsSEXP, SEXP pzSEXP, SEXP sign_switchSEXP, SEXP print_matSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type truth(truthSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type fmeans(fmeansSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type fsigs(fsigsSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type pz(pzSEXP);
    Rcpp::traits::input_parameter< bool >::type sign_switch(sign_switchSEXP);
    Rcpp::traits::input_parameter< bool >::type print_mat(print_matSEXP);
    rcpp_result_gen = Rcpp::wrap(relabel_truth(truth, fmeans, fsigs, pz, sign_switch, print_mat));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_sparsefactor_cavi", (DL_FUNC) &_sparsefactor_cavi, 12},
    {"_sparsefactor_gibbs", (DL_FUNC) &_sparsefactor_gibbs, 10},
    {"_sparsefactor_gibbs_full", (DL_FUNC) &_sparsefactor_gibbs_full, 10},
    {"_sparsefactor_relabel", (DL_FUNC) &_sparsefactor_relabel, 9},
    {"_sparsefactor_relabel_truth", (DL_FUNC) &_sparsefactor_relabel_truth, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_sparsefactor(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
