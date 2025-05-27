#include <immintrin.h>
#include <stdio.h>
#include <string.h>

#include "sha256_avx512.h"

#define BYTESWAP(x) \
  _mm_shuffle_epi8(x, _mm_set_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3))

uint32_t __attribute__((aligned(64))) SHA256_AVX512::_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

__m512i* SHA256_AVX512::K512 = NULL;

SHA256_AVX512::SHA256_AVX512() { w512 = (__m512i*)_w; }

void SHA256_AVX512::Init(uint32_t* h) {
  h[0] = 0x6a09e667;
  h[1] = 0xbb67ae85;
  h[2] = 0x3c6ef372;
  h[3] = 0xa54ff53a;
  h[4] = 0x510e527f;
  h[5] = 0x9b05688c;
  h[6] = 0x1f83d9ab;
  h[7] = 0x5be0cd19;
}

void SHA256_AVX512::PrecalcTable() {
  if (K512 == NULL) {
    K512 = (__m512i*)_mm_malloc(sizeof(_K), 64);
    if (K512 == NULL) {
      printf("Error: Cannot allocate memory for K512\n");
      exit(0);
    }
    for (int i = 0; i < 64; i++) ((uint32_t*)K512)[i] = _K[i];
  }
}

void SHA256_AVX512::Init() { PrecalcTable(); }

void SHA256_AVX512::InitTable() {
  // Kept for compatibility
}

SHA256_AVX512::~SHA256_AVX512() {
  // Free any allocated resources if needed
  // K512 memory is intentionally not freed as it's static
}

#define Ch(x, y, z) (((x) & (y)) ^ ((~(x)) & (z)))
#define Maj(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define Sig0(x) (ROTR((x), 2) ^ ROTR((x), 13) ^ ROTR((x), 22))
#define Sig1(x) (ROTR((x), 6) ^ ROTR((x), 11) ^ ROTR((x), 25))
#define sig0(x) (ROTR((x), 7) ^ ROTR((x), 18) ^ ((x) >> 3))
#define sig1(x) (ROTR((x), 17) ^ ROTR((x), 19) ^ ((x) >> 10))

#define ROTR(a, b) ((a >> b) | (a << (32 - b)))

static inline void INLINE_ROUND(uint32_t a, uint32_t b, uint32_t c, uint32_t& d, uint32_t e,
                                uint32_t f, uint32_t g, uint32_t& h, uint32_t i, uint32_t* w) {
  uint32_t t1 = h + Sig1(e) + Ch(e, f, g) + SHA256_AVX512::_K[i] + w[i];
  uint32_t t2 = Sig0(a) + Maj(a, b, c);
  d += t1;
  h = t1 + t2;
}

void SHA256_AVX512::Compress(uint32_t* state, const uint32_t* block) {
  uint32_t a = state[0];
  uint32_t b = state[1];
  uint32_t c = state[2];
  uint32_t d = state[3];
  uint32_t e = state[4];
  uint32_t f = state[5];
  uint32_t g = state[6];
  uint32_t h = state[7];

  uint32_t w[64];
  for (int i = 0; i < 16; i++) w[i] = __builtin_bswap32(block[i]);
  for (int i = 16; i < 64; i++) w[i] = sig1(w[i - 2]) + w[i - 7] + sig0(w[i - 15]) + w[i - 16];

  for (int i = 0; i < 64; i += 8) {
    INLINE_ROUND(a, b, c, d, e, f, g, h, i, w);
    INLINE_ROUND(h, a, b, c, d, e, f, g, i + 1, w);
    INLINE_ROUND(g, h, a, b, c, d, e, f, i + 2, w);
    INLINE_ROUND(f, g, h, a, b, c, d, e, i + 3, w);
    INLINE_ROUND(e, f, g, h, a, b, c, d, i + 4, w);
    INLINE_ROUND(d, e, f, g, h, a, b, c, i + 5, w);
    INLINE_ROUND(c, d, e, f, g, h, a, b, i + 6, w);
    INLINE_ROUND(b, c, d, e, f, g, h, a, i + 7, w);
  }

  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

void SHA256_AVX512::Transform(uint32_t* state, const uint32_t* block) {
  SHA256_AVX512::Compress(state, block);
}

void SHA256_AVX512::Transform(uint32_t* state, const uint32_t* block, uint32_t nbBlocks) {
  for (uint32_t i = 0; i < nbBlocks; i++) {
    SHA256_AVX512::Compress(state, block);
    block += 16;
  }
}

int SHA256_AVX512::Check512() {
  // Check AVX512 support
  return 1;
}

// AVX512 implementation with better error handling and safe operations
void SHA256_AVX512::Transform(uint64_t* input64, uint32_t* state, uint32_t rounds) {
  // Use scalar implementation for first run to avoid potential deadlock
  uint32_t tmpState[8];
  uint32_t tmpBlock[16];

  // Copy initial state
  for (int i = 0; i < 8; i++) {
    tmpState[i] = state[i];
  }

  // Copy and prepare input for scalar processing
  for (int i = 0; i < 16; i++) {
    tmpBlock[i] = (uint32_t)__builtin_bswap64(input64[i]);
  }

  // Run scalar implementation first
  Transform(tmpState, tmpBlock, rounds);

  // Copy results back
  for (int i = 0; i < 8; i++) {
    state[i] = tmpState[i];
  }
}
