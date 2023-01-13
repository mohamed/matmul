#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <chrono>
#ifdef USE_OPENMP
#include <omp.h>
#endif

template <int rows, int columns, int inners>
inline void matmulImplNaive(const float *left, const float *right,
                            float *result) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      for (int inner = 0; inner < inners; inner++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
} } } }

template <int rows, int columns, int inners>
inline void matmulImplNaiveRegisterAcc(const float *left, const float *right,
                                       float *result) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      float acc = 0.0;
      for (int inner = 0; inner < inners; inner++) {
        acc += left[row * columns + inner] * right[inner * columns + col];
      }
      result[row * columns + col] = acc;
} } }

template <int rows, int columns, int inners>
inline void matmulImplLoopOrder(const float *left, const float *right,
                                float *result) {
  for (int row = 0; row < rows; row++) {
    for (int inner = 0; inner < inners; inner++) {
      for (int col = 0; col < columns; col++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
} } } }

template <int rows, int columns, int inners, int tileSize>
inline void matmulImplTiling(const float *left, const float *right,
                             float *result) {
  for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
    for (int row = 0; row < rows; row++) {
      int innerTileEnd = std::min(inners, innerTile + tileSize);
      for (int inner = innerTile; inner < innerTileEnd; inner++) {
        for (int column = 0; column < columns; column++) {
          result[row * columns + column] +=
              left[row * inners + inner] * right[inner * columns + column];
} } } } }

#ifdef USE_OPENMP
template <int rows, int columns, int inners,
          int tileSize = 16>
inline void matmulImplRowColParallelInnerTiling(const float *left,
                                                const float *right,
                                                float *result) {
#pragma omp parallel for shared(result, left, right) default(none) \
  collapse(2) num_threads(8)
  for (int rowTile = 0; rowTile < rows; rowTile += 256) {
    for (int columnTile = 0; columnTile < columns; columnTile += 256) {
      for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
        for (int row = rowTile; row < rowTile + 256; row++) {
          int innerTileEnd = std::min(inners, innerTile + tileSize);
          for (int inner = innerTile; inner < innerTileEnd; inner++) {
            for (int col = columnTile; col < columnTile + 256; col++) {
              result[row * columns + col] +=
                  left[row * inners + inner] * right[inner * columns + col];
} } } } } } }
#endif

int main() {
  const int R = 1024, C = 1024, I = 512;
  const int T = 10;
  float A[R*I], B[I*C], Z[R*C];
  std::chrono::time_point<std::chrono::high_resolution_clock> begin, end;

  for (int r = 0; r < R; r++) {
    for (int i = 0; i < I; i++) {
      A[r*I + i] = 1.0;
    }
  }
  for (int i = 0; i < I; i++) {
    for (int c = 0; c < C; c++) {
      B[i*C + c] = 2.0;
    }
  }

  begin = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < T; i++) {
    matmulImplNaive<R,C,I>(A, B, Z);
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "matmulImplNaive: " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()/T << "us" << std::endl;

  begin = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < T; i++) {
    matmulImplNaiveRegisterAcc<R,C,I>(A, B, Z);
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "matmulImplNaiveRegisterAcc: " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()/T << "us" << std::endl;

  begin = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < T; i++) {
    matmulImplLoopOrder<R,C,I>(A,B,Z);
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "matmulImplLoopOrder: " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()/T << "us" << std::endl;

  begin = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < T; i++) {
    matmulImplTiling<R,C,I,16>(A,B,Z);
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "matmulImplTiling: " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()/T << "us" << std::endl;

#ifdef USE_OPENMP
  begin = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < T; i++) {
    matmulImplRowColParallelInnerTiling<R,C,I,16>(A,B,Z);
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "matmulImplRowColParallelInnerTiling: " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()/T << "us" << std::endl;
#endif

  return EXIT_SUCCESS;
}
