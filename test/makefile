CXX = g++

SRC_DIR = ../src
INCLUDE_DIRS = -I../libs -I$(SRC_DIR)
LIB_FLAGS = -lm -lpthread

OBJ = $(SRC_DIR)/ssvm_sifs.o

UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
	CXXFLAGS = -Wall -O3 -fPIC -std=c++11 -ftree-vectorize -msse2 -mfpmath=sse $(INCLUDE_DIRS)
else
	CXXFLAGS = -Wall -O3 -fopenmp -fPIC -std=c++11 -ftree-vectorize -msse2 -mfpmath=sse $(INCLUDE_DIRS)
endif

all: start 

start: start.o $(OBJ)
	$(CXX) $(CXXFLAGS) -o start start.o $(OBJ) $(LIB_FLAGS)

.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(SRC_DIR)/.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o $(SRC_DIR)/*.o test_main start
