CXX         = g++
CPPFLAGS   := -std=c++11 -D_FORTIFY_SOURCE=2
CXXFLAGS   := -Wall -Wextra -Werror -Ofast -march=native -fstack-protector-all
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
	ulimit -s unlimited && ./$(OUT)

clean:
	$(RM) $(OUT)
