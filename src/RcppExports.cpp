// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// gibbs
List gibbs(int n_iter, arma::mat& ymat, arma::vec& pivec, double ptaushape, double ptaurate, double palphashape, double palpharate, bool sign_switch, bool label_switch, int seed, bool debug);
RcppExport SEXP _sparsefactor_gibbs(SEXP n_iterSEXP, SEXP ymatSEXP, SEXP pivecSEXP, SEXP ptaushapeSEXP, SEXP ptaurateSEXP, SEXP palphashapeSEXP, SEXP palpharateSEXP, SEXP sign_switchSEXP, SEXP label_switchSEXP, SEXP seedSEXP, SEXP debugSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n_iter(n_iterSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type ymat(ymatSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type pivec(pivecSEXP);
    Rcpp::traits::input_parameter< double >::type ptaushape(ptaushapeSEXP);
    Rcpp::traits::input_parameter< double >::type ptaurate(ptaurateSEXP);
    Rcpp::traits::input_parameter< double >::type palphashape(palphashapeSEXP);
    Rcpp::traits::input_parameter< double >::type palpharate(palpharateSEXP);
    Rcpp::traits::input_parameter< bool >::type sign_switch(sign_switchSEXP);
    Rcpp::traits::input_parameter< bool >::type label_switch(label_switchSEXP);
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< bool >::type debug(debugSEXP);
    rcpp_result_gen = Rcpp::wrap(gibbs(n_iter, ymat, pivec, ptaushape, ptaurate, palphashape, palpharate, sign_switch, label_switch, seed, debug));
    return rcpp_result_gen;
END_RCPP
}
// myFunction
Rcpp::StringVector myFunction(Rcpp::StringVector x);
RcppExport SEXP _sparsefactor_myFunction(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(myFunction(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpp_hello_world
List rcpp_hello_world();
RcppExport SEXP _sparsefactor_rcpp_hello_world() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcpp_hello_world());
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_sparsefactor_gibbs", (DL_FUNC) &_sparsefactor_gibbs, 11},
    {"_sparsefactor_myFunction", (DL_FUNC) &_sparsefactor_myFunction, 1},
    {"_sparsefactor_rcpp_hello_world", (DL_FUNC) &_sparsefactor_rcpp_hello_world, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_sparsefactor(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
