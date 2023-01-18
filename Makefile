CXX         = g++
CPPFLAGS   := -std=c++11
CXXFLAGS   := -Wall -Wextra -Werror -O3 -march=native -ffast-math
LDFLAGS    := -static-libgcc -static-libstdc++
ifdef USE_OPENMP
CPPFLAGS   += -DUSE_OPENMP
LDFLAGS    += -fopenmp
endif
SRCS        = matmul.cc
OUT         = matmul.x

all: $(OUT)

$(OUT): $(SRCS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run: $(OUT)
	ulimit -s unlimited
	./$(OUT)

clean:
	$(RM) $(OUT)
