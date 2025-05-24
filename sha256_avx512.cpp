#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "sha256_avx512.h"

namespace _sha256avx512 {

#define ALIGN64 __attribute__((aligned(64)))

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

// SHA-256 initial hash values
static const ALIGN64 uint32_t initConstants[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                                  0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

#define SHR(x, n) _mm512_srli_epi32(x, n)
#define ROTR(x, n) _mm512_or_epi32(SHR(x, n), _mm512_slli_epi32(x, 32 - n))
#define CH(x, y, z) _mm512_xor_epi32(_mm512_and_epi32(x, y), _mm512_andnot_epi32(x, z))
#define MAJ(x, y, z) _mm512_ternarylogic_epi32(x, y, z, 0xE8)
#define EP0(x) (_mm512_xor_epi32(_mm512_xor_epi32(ROTR(x, 2), ROTR(x, 13)), ROTR(x, 22)))
#define EP1(x) (_mm512_xor_epi32(_mm512_xor_epi32(ROTR(x, 6), ROTR(x, 11)), ROTR(x, 25)))
#define SIG0(x) (_mm512_xor_epi32(_mm512_xor_epi32(ROTR(x, 7), ROTR(x, 18)), SHR(x, 3)))
#define SIG1(x) (_mm512_xor_epi32(_mm512_xor_epi32(ROTR(x, 17), ROTR(x, 19)), SHR(x, 10)))

void Initialize(__m512i state[8]) {
  for (int i = 0; i < 8; ++i) state[i] = _mm512_set1_epi32(initConstants[i]);
}

void Transform(__m512i state[8], const uint8_t* data[32]) {
  __m512i a = state[0], b = state[1], c = state[2], d = state[3];
  __m512i e = state[4], f = state[5], g = state[6], h = state[7];
  __m512i W[64], T1, T2;

  for (int t = 0; t < 16; ++t) {
    uint32_t wt[32];
    for (int lane = 0; lane < 32; ++lane) {
      const uint8_t* ptr = data[lane] + t * 4;
      wt[lane] =
          (uint32_t(ptr[0]) << 24) | (uint32_t(ptr[1]) << 16) | (uint32_t(ptr[2]) << 8) | ptr[3];
    }
    W[t] = _mm512_loadu_epi32(wt);
  }

  for (int t = 16; t < 64; ++t) {
    W[t] = _mm512_add_epi32(_mm512_add_epi32(SIG1(W[t - 2]), W[t - 7]),
                            _mm512_add_epi32(SIG0(W[t - 15]), W[t - 16]));
  }

  for (int t = 0; t < 64; ++t) {
    T1 = _mm512_add_epi32(_mm512_add_epi32(h, EP1(e)),
                          _mm512_add_epi32(CH(e, f, g), _mm512_set1_epi32(K[t])));
    T1 = _mm512_add_epi32(T1, W[t]);
    T2 = _mm512_add_epi32(EP0(a), MAJ(a, b, c));

    h = g;
    g = f;
    f = e;
    e = _mm512_add_epi32(d, T1);
    d = c;
    c = b;
    b = a;
    a = _mm512_add_epi32(T1, T2);
  }

  state[0] = _mm512_add_epi32(state[0], a);
  state[1] = _mm512_add_epi32(state[1], b);
  state[2] = _mm512_add_epi32(state[2], c);
  state[3] = _mm512_add_epi32(state[3], d);
  state[4] = _mm512_add_epi32(state[4], e);
  state[5] = _mm512_add_epi32(state[5], f);
  state[6] = _mm512_add_epi32(state[6], g);
  state[7] = _mm512_add_epi32(state[7], h);
}

}  // namespace _sha256avx512

extern "C" void sha256_avx512_32blocks(const uint8_t* data[32], uint8_t* out[32]) {
  __m512i state[8];
  _sha256avx512::Initialize(state);
  _sha256avx512::Transform(state, data);

  alignas(64) uint32_t digest[8][32];
  for (int i = 0; i < 8; ++i) _mm512_store_si512((__m512i*)digest[i], state[i]);

  for (int lane = 0; lane < 32; ++lane) {
    for (int i = 0; i < 8; ++i) {
      uint32_t word = digest[i][lane];
#if defined(_MSC_VER)
      word = _byteswap_ulong(word);
#else
      word = __builtin_bswap32(word);
#endif
      memcpy(out[lane] + i * 4, &word, 4);
    }
  }
}

template <int N>
static void batch_pad_and_run(const uint8_t* const in[N], uint8_t* const out[N]) {
  alignas(64) static const uint8_t zero_block[64] = {0};
  alignas(64) static uint8_t dummy_out[64] = {0};

  const uint8_t* padded_in[32];
  uint8_t* padded_out[32];

  for (int i = 0; i < N; ++i) {
    padded_in[i] = in[i];
    padded_out[i] = out[i];
  }
  for (int i = N; i < 32; ++i) {
    padded_in[i] = zero_block;
    padded_out[i] = dummy_out;
  }

  sha256_avx512_32blocks(padded_in, padded_out);
}

extern "C" void sha256_avx512_16blocks(const uint8_t* in[16], uint8_t* out[16]) {
  batch_pad_and_run<16>(in, out);
}
extern "C" void sha256_avx512_8blocks(const uint8_t* in[8], uint8_t* out[8]) {
  batch_pad_and_run<8>(in, out);
}
extern "C" void sha256_avx512_4blocks(const uint8_t* in[4], uint8_t* out[4]) {
  batch_pad_and_run<4>(in, out);
}
extern "C" void sha256_avx512_1block(const uint8_t* in, uint8_t* out) {
  const uint8_t* in_[1] = {in};
  uint8_t* out_[1] = {out};
  batch_pad_and_run<1>(in_, out_);
}
