#ifndef LAP_H
#define LAP_H

#define LARGE 1e300
#define NEW(x, t, n) if ((x = (t *)malloc(sizeof(t) * (n))) == 0) { return -1; }
#define FREE(x) if (x != 0) { free(x); x = 0; }
#define SWAP_INDICES(a, b) { int_t _temp_index = a; a = b; b = _temp_index; }

typedef signed int int_t;
typedef unsigned int uint_t;
typedef double cost_t;

int_t lapjv_internal(const uint_t n, cost_t *cost[], int_t *x, int_t *y);

#endif
