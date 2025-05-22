#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "sha256_avx512.h"

namespace _sha256avx512 {

#define ALIGN64 __attribute__((aligned(64)))

// Initial hash values for 32 parallel lanes (explicitly unrolled)
static const ALIGN64 uint32_t initConstants[32][8] = {
    // Lane 0
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 1
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 2
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 3
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 4
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 5
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 6
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 7
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 8
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 9
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 10
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 11
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 12
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 13
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 14
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 15
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 16
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 17
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 18
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 19
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 20
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 21
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 22
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 23
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 24
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 25
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 26
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 27
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 28
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 29
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 30
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19},
    // Lane 31
    {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
     0x5be0cd19}};

// SHA-256 round constants
static const ALIGN64 uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// Vectorized SHA-256 macros
#define SHR(x, n) _mm512_srli_epi32(x, n)
#define ROTR(x, n) _mm512_or_si512(SHR(x, n), _mm512_slli_epi32(x, 32 - n))
#define CH(x, y, z) _mm512_xor_si512(_mm512_and_si512(x, y), _mm512_andnot_si512(x, z))
#define MAJ(x, y, z) \
  _mm512_or_si512(_mm512_and_si512(x, y), _mm512_and_si512(z, _mm512_or_si512(x, y)))

#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ SHR(x, 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ SHR(x, 10))

void Initialize(__m512i state[8]) {
  for (int i = 0; i < 8; i++) {
    state[i] = _mm512_load_epi32(initConstants[i]);
  }
}

void Transform(__m512i state[8], const uint8_t* data[32]) {
  __m512i a[8], b[8];
  __m512i W[64];
  __m512i T1, T2;

  // Load state
  for (int i = 0; i < 8; i++) a[i] = state[i];

  // Message schedule
  for (int t = 0; t < 16; t++) {
    ALIGN64 uint32_t wt[32];
    for (int lane = 0; lane < 32; lane++) {
      const uint8_t* ptr = data[lane] + t * 4;
      wt[lane] = (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
    }
    W[t] = _mm512_load_epi32(wt);
  }

  for (int t = 16; t < 64; t++) {
    W[t] = _mm512_add_epi32(_mm512_add_epi32(SIG1(W[t - 2]), W[t - 7]),
                            _mm512_add_epi32(SIG0(W[t - 15]), W[t - 16]));
  }

  // Main compression loop
  for (int t = 0; t < 64; t++) {
    T1 = _mm512_add_epi32(
        _mm512_add_epi32(_mm512_add_epi32(a[7], EP1(a[4])),
                         _mm512_add_epi32(CH(a[4], a[5], a[6]), _mm512_set1_epi32(K[t]))),
        W[t]);

    T2 = _mm512_add_epi32(EP0(a[0]), MAJ(a[0], a[1], a[2]));

    // Rotate registers
    a[7] = a[6];
    a[6] = a[5];
    a[5] = a[4];
    a[4] = _mm512_add_epi32(a[3], T1);
    a[3] = a[2];
    a[2] = a[1];
    a[1] = a[0];
    a[0] = _mm512_add_epi32(T1, T2);
  }

  // Update state
  for (int i = 0; i < 8; i++) {
    state[i] = _mm512_add_epi32(state[i], a[i]);
  }
}

}  // namespace _sha256avx512

// Public API function
void sha256_avx512_32blocks(const uint8_t* data[32], uint8_t* out_hashes[32]) {
  __m512i state[8];
  ALIGN64 uint32_t digest[32][8];

  _sha256avx512::Initialize(state);
  _sha256avx512::Transform(state, data);

  // Store results
  for (int lane = 0; lane < 32; lane++) {
    for (int i = 0; i < 8; i++) {
      uint32_t val = _mm512_extract_epi32(state[i], lane);
      val = __builtin_bswap32(val);
      memcpy(out_hashes[lane] + i * 4, &val, 4);
    }
  }
}