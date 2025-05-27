#ifndef SHA256_AVX512_H
#define SHA256_AVX512_H

#include <immintrin.h>
#include <stdint.h>

class SHA256_AVX512 {
 public:
  SHA256_AVX512();
  ~SHA256_AVX512();
  void Init();
  void InitTable();
  int Check512();
  void Transform(uint64_t* input64, uint32_t* state, uint32_t rounds = 1);
  static void PrecalcTable();

  // Static SHA256 functions from original
  static void Init(uint32_t* h);
  static void Transform(uint32_t* state, const uint32_t* block);
  static void Transform(uint32_t* state, const uint32_t* block, uint32_t nbBlocks);
  static void Compress(uint32_t* state, const uint32_t* block);
  static uint32_t _K[64];

 private:
  __m512i _w512[16];
  uint32_t __attribute__((aligned(64))) _w[16][16];
  __m512i* __attribute__((aligned(64))) w512;
  static __m512i* K512;
};

#endif  // SHA256_AVX512_H
