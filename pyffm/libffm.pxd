cimport numpy as np

cdef extern from "ffm.h" namespace "ffm":
    
    cdef struct ffm_node
    cdef struct ffm_problem
    cdef struct ffm_model
    cdef struct ffm_parameter
    
    void ffm_destroy_problem(ffm_problem **)
    void ffm_destroy_model(ffm_model **)
    ffm_parameter ffm_get_default_param()
    ffm_model *ffm_train(ffm_problem *, ffm_parameter) nogil
    float ffm_predict(ffm_node *, ffm_node *, ffm_model *)