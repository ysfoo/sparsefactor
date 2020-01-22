#include <Rcpp.h>
#include "modString.h"

//' @title myFunction
//' @description
//' Modify a string in Rcpp.
//' @name myFunction
//' @param x a vector of strings
//' @examples
//' myFunction(x=c('Hello', "C++", 'header', 'files'))
//'
//' @export
// [[Rcpp::export]]
Rcpp::StringVector myFunction(Rcpp::StringVector x) {
  x = modString(x);
  return x;
}
