#include <stdlib.h>
#include <numpy/arrayobject.h>
#include "ffm.h"

/*
 * Modified from libsvm_helper.c in sklearn
 */

/*
 * Convert matrix to sparse representation suitable for libffm. x is
 * expected to be an array of length nrow*ncol.
 *
 * Typically the matrix will be dense, so we speed up the routine for
 * this case. We create a temporary array temp that collects non-zero
 * elements and after we just memcpy that to the proper array.
 *
 * Special care must be taken with indinces, since libffm indices start
 * at 1 and not at 0. (TODO: check)
 *
 * Strictly speaking, the C standard does not require that structs are
 * contiguous, but in practice its a reasonable assumption.
 *
 */

struct ffm_node *dense_to_libffm (double *x, npy_intp *dims, int *n, int *m)
{
    struct ffm_node *node;
    npy_intp len_row = dims[1];
    double *tx = x;
    int i;

    node = malloc (dims[0] * sizeof(struct ffm_node));

    if (node == NULL) return NULL;
    for (i=0; i<dims[0]; ++i) {
        node[i].values = tx;
        node[i].dim = (int) len_row;
        node[i].ind = i; /* only used if kernel=precomputed, but not
                            too much overhead */
        tx += len_row;
    }

    return node;
}

/*
 * Fill an ffm_parameter struct.
 */
void set_parameter(struct ffm_parameter *param, ffm_float eta, ffm_float lambda,
                   ffm_int nr_iters, ffm_int k, ffm_int nr_threads, bool quiet,
                   bool normalization, bool random)
{
    param->eta = eta;
    param->lambda = lambda;
    param->nr_iters = nr_iters;
    param->k = k;
    param->nr_threads = nr_threads;
    param->quiet = quiet;
    param->normalization = normalization;
    param->random = random;
}

/*
 * Fill an ffm_problem struct. problem->x will be malloc'd.
 */
void set_problem(struct ffm_problem *problem, char *X, char *Y, char *sample_weight, npy_intp *dims, int kernel_type)
{
    if (problem == NULL) return;
    problem->l = (int) dims[0]; /* number of samples */
    problem->n = 0
    problem->m = 0
    problem->Y = (double *) Y;

    dense_to_libffm((double *) X, dims, problem.n, problem.m); /* implicit call to malloc */
    
}