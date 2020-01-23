#ifndef RELABEL_H
#define RELABEL_H

#include <RcppArmadillo.h>

#define NEW(x, t, n) if ((x = (t *)malloc(sizeof(t) * (n))) == 0) { return -1; }
#define FREE(x) if (x != 0) { free(x); x = 0; }
#define SWAP_INDICES(a, b) { int_t _temp_index = a; a = b; b = _temp_index; }

typedef signed int int_t;
typedef unsigned int uint_t;
typedef double cost_t;

using namespace Rcpp;

List relabel(List samples, bool sign_switch, bool label_switch);

#endif
