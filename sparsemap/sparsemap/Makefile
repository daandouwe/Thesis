DYNET_PATH ?= ${HOME}/dynet-base/dynet
EIGEN_PATH ?= ${HOME}/dynet-base/eigen
AD3_PATH ?= ${HOME}/AD3

CC = g++
DEBUG =
INCLUDES = -I$(DYNET_PATH) -I$(EIGEN_PATH) -I$(AD3_PATH)
LIBS = -L$(AD3_PATH)/ad3
CFLAGS = -O3 -Wall -Wno-sign-compare -Wno-int-in-bool-context -c -fmessage-length=0 $(INCLUDES) -DEIGEN_FAST_MATH -fPIC -fno-finite-math-only -Wno-missing-braces -std=c++11 -funroll-loops $(DEBUG)
LFLAGS = -ldynet -lad3

ifdef MKL_PATH
INCLUDES += -I$(MKL_PATH)/include
LIBS += -L$(MKL_PATH)/lib/intel64
LFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
CFLAGS += -DEIGEN_USE_MKL_ALL
endif

LIBS_CPU = $(LIBS) -L$(DYNET_PATH)/build/dynet/
LFLAGS_CPU = $(LIBS_CPU) $(LFLAGS)

FactorTree.o: FactorTree.cc FactorTree.h
	$(CC) $(CFLAGS) -L${AD3_PATH}/ad3/ FactorTree.cc

sparseparse.o : sparseparse.cc sparseparse.h FactorTree.h
	$(CC) $(CFLAGS) sparseparse.cc -o sparseparse.o

test_sparseparse: sparseparse.o test_sparseparse.o FactorTree.o
	$(CC) FactorTree.o sparseparse.o test_sparseparse.o $(LFLAGS_CPU) -o test_sparseparse

test_sparseparse.o : test_sparseparse.cc sparseparse.h
	$(CC) $(CFLAGS) test_sparseparse.cc

clean:
	rm -f *.o *~ test_sparseparse
