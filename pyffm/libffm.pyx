import  numpy as np
cimport numpy as np

np.import_array()

def train_wrap(np.ndarray[np.float64_t, ndim=2, mode='c'] X, 
               np.ndarray[np.float64_t, ndim=1, mode='c'] Y,
               double eta=0.1, double lam=0.0, int nr_iters=15, int k=4,
               nr_threads=8, quiet=False, normalization=True, random=True):
    cdef ffm_parameter parameters
    cdef ffm_problem * prob
    parameters.eta = eta
    parameters.lambda = lam
    parameters.nr_iters = nr_iters
    parameters.k = k
    parameters.nr_threads = nr_threads
    parameters.quiet = quiet
    parameters.normalization = normalization
    parameters.random = random
    
    cdef ffm_model * model = ffm_train(problem, parameters)
    n = model->n
    m = model->m
    k = model->k
    norm = model->normalization