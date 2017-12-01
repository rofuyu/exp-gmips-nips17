CXX?=g++
#CXXFLAGS=-fopenmp -static -O3
CXXFLAGS=-fopenmp -ffast-math -pipe -g -fPIC
CXXFLAGS=-fopenmp -march=native -ffast-math -fPIC -g
CXXFLAGS=-fopenmp -march=native -ffast-math -O3 -std=c++11
CXXFLAGS=-fopenmp -march=native -ffast-math -std=c++11 -O3 -I ff 
CXXFLAGS=-fopenmp -march=native -O3 -std=c++11 -fPIC 
VERSION=0.1

#CXXFLAGS+= -D_USE_FLOAT_

BLASFLAGS=-llapack_atlas -lf77blas -lcblas -latlas -lgfortran -L/u/rofuyu/.local/lib  #GCC in UTCS
BLASFLAGS=-llapack -lptf77blas -lptcblas -latlas -L/home/01845/rofuyu/.local/lib  # GCC in TACC
BUILD_BLASFLAGS=-llapack -lptf77blas -lptcblas -latlas -L/home/01845/rofuyu/.local/lib  # GCC in TACC
BUILD_BLASFLAGS=-llapack -lcblas -latlas -L/home/01845/rofuyu/.local/lib  # GCC in TACC
BLASFLAGS=-lblas -llapack
LIBSFLAGS=$(BLASFLAGS) 

all: go _tclib.so

go: go.cpp fast-pred.h sparse_matrix.h pmf.h
	${CXX} ${CXXFLAGS} -o go go.cpp ${BLASFLAGS}

_tclib.so: _tclib.cpp fast-pred.h pmf.o
	$(CXX) -fPIC -fopenmp $(CXXFLAGS) -shared -Wl,-soname,_tclib.so -o _tclib.so -lgomp _tclib.cpp pmf.o

pmf.o: pmf.cpp pmf.h sparse_matrix.h
	${CXX} ${CXXFLAGS} -c -o pmf.o pmf.cpp

zip:
	make clean; cd ../;  zip -r --symlinks fast-pred-${VERSION}.zip fast-pred-${VERSION}/

clean:
	rm -rf go pmf.o *.so *~ .*~

