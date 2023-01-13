CXX         = g++
CPPFLAGS   := -DUSE_OPENMP
CXXFLAGS   := -std=c++11 -Wall -Wextra -Werror -O3 -march=native -ffast-math -fopenmp
LDFLAGS    :=
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
