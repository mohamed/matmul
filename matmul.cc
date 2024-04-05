#include <iostream>
#include <cstdlib>
#include <cmath>
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
  for (int I = 0; I < rows; I+=tileSize) {
    for (int J = 0; J < columns; J+=tileSize) {
      for (int K = 0; K < inners; K+=tileSize) {

        for (int i = 0; i < tileSize; i++) {
          for (int j = 0; j < tileSize; j++) {
            for (int k = 0; k < tileSize; k++) {
              result[(I + i) * columns + (J + j)] += \
                left[(I + i) * inners + (K + k)] * \
                right[(K + k) * columns + (J + j)];
            }
          }
        }
      }
    }
  }
}

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

template <int rows, int columns>
bool verify(const float * result, const float * gold) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (fabs(result[i * columns + j] - gold[i * columns + j]) > 0.001) {
        printf("Mismatch: %f\n", result[i * columns + j]);
        return false;
      }
    }
  }
  return true;
}

int main() {
  const int R = 128, C = 256, I = 512;
  const int T = 1;
  float A[R*I], B[I*C], Z[R*C], gold[R*C];
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
    memset(gold, 0.0, R*C*sizeof(float));
    matmulImplNaive<R,C,I>(A, B, gold);
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "matmulImplNaive: " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()/T << "us" << std::endl;
  if (!verify<R,C>(gold, gold)) abort();

  begin = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < T; i++) {
    memset(Z, 0.0, R*C*sizeof(float));
    matmulImplNaiveRegisterAcc<R,C,I>(A, B, Z);
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "matmulImplNaiveRegisterAcc: " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()/T << "us" << std::endl;
  if (!verify<R,C>(Z, gold)) abort();

  begin = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < T; i++) {
    memset(Z, 0.0, R*C*sizeof(float));
    matmulImplLoopOrder<R,C,I>(A,B,Z);
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "matmulImplLoopOrder: " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()/T << "us" << std::endl;
  if (!verify<R,C>(Z, gold)) abort();

  begin = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < T; i++) {
    memset(Z, 0.0, R*C*sizeof(float));
    matmulImplTiling<R,C,I,16>(A,B,Z);
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "matmulImplTiling: " << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()/T << "us" << std::endl;
  if (!verify<R,C>(Z, gold)) abort();

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
