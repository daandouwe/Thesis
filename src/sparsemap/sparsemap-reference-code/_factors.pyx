from libcpp cimport bool
from libcpp.vector cimport vector

from ad3.base cimport Factor, GenericFactor, PGenericFactor


cdef extern from "FactorMatching.h" namespace "sparsemap":
    cdef cppclass FactorMatching(Factor):
        FactorMatching()
        void Initialize(int, int)


cdef class PFactorMatching(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
            self.thisptr = new FactorMatching()

    def __dealloc(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int rows, int cols):
        (<FactorMatching*>self.thisptr).Initialize(rows, cols)
