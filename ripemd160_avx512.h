#ifndef RIPEMD160_AVX512_H
#define RIPEMD160_AVX512_H

#include <immintrin.h>
#include <stdint.h>

class RIPEMD160_AVX512 {
 public:
  RIPEMD160_AVX512();
  ~RIPEMD160_AVX512();
  void Init();
  int Check512();
  void Transform(uint64_t* input64, uint32_t* state, uint32_t rounds = 1);

 private:
  __m512i* __attribute__((aligned(64))) w512;
  uint32_t __attribute__((aligned(64))) _w[16][16];
};

#endif  // RIPEMD160_AVX512_H
