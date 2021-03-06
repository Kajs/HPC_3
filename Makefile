TARGET	= libmatmult.so
LIBSRCS	= 
LIBOBJS	= matmult.o

OPT	= -g
PIC = -fpic
XPIC  = -Xcompiler -fpic
XOPTS = -G -Xptxas=-v
ARCH  = -arch=sm_30
OMP   = -fopenmp

CXX	= nvcc
CXXFLAGS= --compiler-options "$(OPT) $(PIC) $(OMP)" $(ARCH) $(XOPTS) $(XPIC)

CUDA_PATH ?= /appl/cuda/8.0
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc

SOFLAGS = -shared
XLIBS	= -lcublas

$(TARGET): $(LIBOBJS)
	$(CXX) -o $@ $(CXXFLAGS) $(SOFLAGS) $(INCLUDES) $^ $(XLIBS)

.SUFFIXES: .cu
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f $(TARGET) $(LIBOBJS) 
