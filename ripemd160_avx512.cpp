#include <immintrin.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "ripemd160_avx512.h"

static const uint64_t sizedesc_64 = 64ULL << 3;
static const unsigned char pad[128] = {0x80};

#define _mm512_not_si512(x) _mm512_xor_si512((x), _mm512_set1_epi32(-1))
#define ROL(x, n) _mm512_or_si512(_mm512_slli_epi32(x, n), _mm512_srli_epi32(x, 32 - n))
#define f1(x, y, z) _mm512_xor_si512(x, _mm512_xor_si512(y, z))
#define f2(x, y, z) _mm512_or_si512(_mm512_and_si512(x, y), _mm512_andnot_si512(x, z))
#define f3(x, y, z) _mm512_xor_si512(_mm512_or_si512(x, _mm512_not_si512(y)), z)
#define f4(x, y, z) _mm512_or_si512(_mm512_and_si512(x, z), _mm512_andnot_si512(z, y))
#define f5(x, y, z) _mm512_xor_si512(x, _mm512_or_si512(y, _mm512_not_si512(z)))
#define add3(x0, x1, x2) _mm512_add_epi32(_mm512_add_epi32(x0, x1), x2)
#define add4(x0, x1, x2, x3) _mm512_add_epi32(_mm512_add_epi32(x0, x1), _mm512_add_epi32(x2, x3))

#define Round(a, b, c, d, e, f, x, k, r)   \
  u = add4(a, f, x, _mm512_set1_epi32(k)); \
  a = _mm512_add_epi32(ROL(u, r), e);      \
  c = ROL(c, 10);

#define R11(a, b, c, d, e, x, r) Round(a, b, c, d, e, f1(b, c, d), x, 0, r)
#define R21(a, b, c, d, e, x, r) Round(a, b, c, d, e, f2(b, c, d), x, 0x5A827999, r)
#define R31(a, b, c, d, e, x, r) Round(a, b, c, d, e, f3(b, c, d), x, 0x6ED9EBA1, r)
#define R41(a, b, c, d, e, x, r) Round(a, b, c, d, e, f4(b, c, d), x, 0x8F1BBCDC, r)
#define R51(a, b, c, d, e, x, r) Round(a, b, c, d, e, f5(b, c, d), x, 0xA953FD4E, r)
#define R12(a, b, c, d, e, x, r) Round(a, b, c, d, e, f5(b, c, d), x, 0x50A28BE6, r)
#define R22(a, b, c, d, e, x, r) Round(a, b, c, d, e, f4(b, c, d), x, 0x5C4DD124, r)
#define R32(a, b, c, d, e, x, r) Round(a, b, c, d, e, f3(b, c, d), x, 0x6D703EF3, r)
#define R42(a, b, c, d, e, x, r) Round(a, b, c, d, e, f2(b, c, d), x, 0x7A6D76E9, r)
#define R52(a, b, c, d, e, x, r) Round(a, b, c, d, e, f1(b, c, d), x, 0, r)

// Safe version using union to avoid strict-aliasing violations
#define LOADW(i)                                                                                   \
  ({                                                                                               \
    union {                                                                                        \
      uint32_t u32;                                                                                \
      uint8_t u8[4];                                                                               \
    } conv[16];                                                                                    \
    for (int j = 0; j < 16; ++j) {                                                                 \
      if (blk[j]) {                                                                                \
        std::memcpy(conv[j].u8, blk[j] + (i * 4), 4);                                              \
      } else {                                                                                     \
        conv[j].u32 = 0;                                                                           \
      }                                                                                            \
    }                                                                                              \
    _mm512_set_epi32(conv[0].u32, conv[1].u32, conv[2].u32, conv[3].u32, conv[4].u32, conv[5].u32, \
                     conv[6].u32, conv[7].u32, conv[8].u32, conv[9].u32, conv[10].u32,             \
                     conv[11].u32, conv[12].u32, conv[13].u32, conv[14].u32, conv[15].u32);        \
  })

// Fixed DEPACK - extract from each of the 5 state vectors with bounds checking
#define DEPACK(d, i)                                  \
  do {                                                \
    if (d && i >= 0 && i < 16) {                      \
      union {                                         \
        uint32_t u32[16];                             \
        __m512i vec;                                  \
      } temp;                                         \
      temp.vec = s[0];                                \
      std::memcpy((uint8_t*)d + 0, &temp.u32[i], 4);  \
      temp.vec = s[1];                                \
      std::memcpy((uint8_t*)d + 4, &temp.u32[i], 4);  \
      temp.vec = s[2];                                \
      std::memcpy((uint8_t*)d + 8, &temp.u32[i], 4);  \
      temp.vec = s[3];                                \
      std::memcpy((uint8_t*)d + 12, &temp.u32[i], 4); \
      temp.vec = s[4];                                \
      std::memcpy((uint8_t*)d + 16, &temp.u32[i], 4); \
    }                                                 \
  } while (0)

static void Initialize(__m512i* s) {
  if (!s) return;

  alignas(64) static const uint32_t init[] = {
      // 16xA
      0x67452301, 0x67452301, 0x67452301, 0x67452301, 0x67452301, 0x67452301, 0x67452301,
      0x67452301, 0x67452301, 0x67452301, 0x67452301, 0x67452301, 0x67452301, 0x67452301,
      0x67452301, 0x67452301,
      // 16xB
      0xEFCDAB89, 0xEFCDAB89, 0xEFCDAB89, 0xEFCDAB89, 0xEFCDAB89, 0xEFCDAB89, 0xEFCDAB89,
      0xEFCDAB89, 0xEFCDAB89, 0xEFCDAB89, 0xEFCDAB89, 0xEFCDAB89, 0xEFCDAB89, 0xEFCDAB89,
      0xEFCDAB89, 0xEFCDAB89,
      // 16xC
      0x98BADCFE, 0x98BADCFE, 0x98BADCFE, 0x98BADCFE, 0x98BADCFE, 0x98BADCFE, 0x98BADCFE,
      0x98BADCFE, 0x98BADCFE, 0x98BADCFE, 0x98BADCFE, 0x98BADCFE, 0x98BADCFE, 0x98BADCFE,
      0x98BADCFE, 0x98BADCFE,
      // 16xD
      0x10325476, 0x10325476, 0x10325476, 0x10325476, 0x10325476, 0x10325476, 0x10325476,
      0x10325476, 0x10325476, 0x10325476, 0x10325476, 0x10325476, 0x10325476, 0x10325476,
      0x10325476, 0x10325476,
      // 16xE
      0xC3D2E1F0, 0xC3D2E1F0, 0xC3D2E1F0, 0xC3D2E1F0, 0xC3D2E1F0, 0xC3D2E1F0, 0xC3D2E1F0,
      0xC3D2E1F0, 0xC3D2E1F0, 0xC3D2E1F0, 0xC3D2E1F0, 0xC3D2E1F0, 0xC3D2E1F0, 0xC3D2E1F0,
      0xC3D2E1F0, 0xC3D2E1F0};
  std::memcpy(s, init, sizeof(init));
}

static void Transform(__m512i* s, uint8_t* blk[16]) {
  if (!s || !blk) return;

  __m512i a1 = _mm512_load_si512(s + 0);
  __m512i b1 = _mm512_load_si512(s + 1);
  __m512i c1 = _mm512_load_si512(s + 2);
  __m512i d1 = _mm512_load_si512(s + 3);
  __m512i e1 = _mm512_load_si512(s + 4);
  __m512i a2 = a1, b2 = b1, c2 = c1, d2 = d1, e2 = e1;
  __m512i u;
  __m512i w[16];

  for (int i = 0; i < 16; ++i) w[i] = LOADW(i);

  R11(a1, b1, c1, d1, e1, w[0], 11);
  R12(a2, b2, c2, d2, e2, w[5], 8);
  R11(e1, a1, b1, c1, d1, w[1], 14);
  R12(e2, a2, b2, c2, d2, w[14], 9);
  R11(d1, e1, a1, b1, c1, w[2], 15);
  R12(d2, e2, a2, b2, c2, w[7], 9);
  R11(c1, d1, e1, a1, b1, w[3], 12);
  R12(c2, d2, e2, a2, b2, w[0], 11);
  R11(b1, c1, d1, e1, a1, w[4], 5);
  R12(b2, c2, d2, e2, a2, w[9], 13);
  R11(a1, b1, c1, d1, e1, w[5], 8);
  R12(a2, b2, c2, d2, e2, w[2], 15);
  R11(e1, a1, b1, c1, d1, w[6], 7);
  R12(e2, a2, b2, c2, d2, w[11], 15);
  R11(d1, e1, a1, b1, c1, w[7], 9);
  R12(d2, e2, a2, b2, c2, w[4], 5);
  R11(c1, d1, e1, a1, b1, w[8], 11);
  R12(c2, d2, e2, a2, b2, w[13], 7);
  R11(b1, c1, d1, e1, a1, w[9], 13);
  R12(b2, c2, d2, e2, a2, w[6], 7);
  R11(a1, b1, c1, d1, e1, w[10], 14);
  R12(a2, b2, c2, d2, e2, w[15], 8);
  R11(e1, a1, b1, c1, d1, w[11], 15);
  R12(e2, a2, b2, c2, d2, w[8], 11);
  R11(d1, e1, a1, b1, c1, w[12], 6);
  R12(d2, e2, a2, b2, c2, w[1], 14);
  R11(c1, d1, e1, a1, b1, w[13], 7);
  R12(c2, d2, e2, a2, b2, w[10], 14);
  R11(b1, c1, d1, e1, a1, w[14], 9);
  R12(b2, c2, d2, e2, a2, w[3], 12);
  R11(a1, b1, c1, d1, e1, w[15], 8);
  R12(a2, b2, c2, d2, e2, w[12], 6);

  R21(e1, a1, b1, c1, d1, w[7], 7);
  R22(e2, a2, b2, c2, d2, w[6], 9);
  R21(d1, e1, a1, b1, c1, w[4], 6);
  R22(d2, e2, a2, b2, c2, w[11], 13);
  R21(c1, d1, e1, a1, b1, w[13], 8);
  R22(c2, d2, e2, a2, b2, w[3], 15);
  R21(b1, c1, d1, e1, a1, w[1], 13);
  R22(b2, c2, d2, e2, a2, w[7], 7);
  R21(a1, b1, c1, d1, e1, w[10], 11);
  R22(a2, b2, c2, d2, e2, w[0], 12);
  R21(e1, a1, b1, c1, d1, w[6], 9);
  R22(e2, a2, b2, c2, d2, w[13], 8);
  R21(d1, e1, a1, b1, c1, w[15], 7);
  R22(d2, e2, a2, b2, c2, w[5], 9);
  R21(c1, d1, e1, a1, b1, w[3], 15);
  R22(c2, d2, e2, a2, b2, w[10], 11);
  R21(b1, c1, d1, e1, a1, w[12], 7);
  R22(b2, c2, d2, e2, a2, w[14], 7);
  R21(a1, b1, c1, d1, e1, w[0], 12);
  R22(a2, b2, c2, d2, e2, w[15], 7);
  R21(e1, a1, b1, c1, d1, w[9], 15);
  R22(e2, a2, b2, c2, d2, w[8], 12);
  R21(d1, e1, a1, b1, c1, w[5], 9);
  R22(d2, e2, a2, b2, c2, w[12], 7);
  R21(c1, d1, e1, a1, b1, w[2], 11);
  R22(c2, d2, e2, a2, b2, w[4], 6);
  R21(b1, c1, d1, e1, a1, w[14], 7);
  R22(b2, c2, d2, e2, a2, w[9], 15);
  R21(a1, b1, c1, d1, e1, w[11], 13);
  R22(a2, b2, c2, d2, e2, w[1], 13);
  R21(e1, a1, b1, c1, d1, w[8], 12);
  R22(e2, a2, b2, c2, d2, w[2], 11);

  R31(d1, e1, a1, b1, c1, w[3], 11);
  R32(d2, e2, a2, b2, c2, w[15], 9);
  R31(c1, d1, e1, a1, b1, w[10], 13);
  R32(c2, d2, e2, a2, b2, w[5], 7);
  R31(b1, c1, d1, e1, a1, w[14], 6);
  R32(b2, c2, d2, e2, a2, w[1], 15);
  R31(a1, b1, c1, d1, e1, w[4], 7);
  R32(a2, b2, c2, d2, e2, w[3], 11);
  R31(e1, a1, b1, c1, d1, w[9], 14);
  R32(e2, a2, b2, c2, d2, w[7], 8);
  R31(d1, e1, a1, b1, c1, w[15], 9);
  R32(d2, e2, a2, b2, c2, w[14], 6);
  R31(c1, d1, e1, a1, b1, w[8], 13);
  R32(c2, d2, e2, a2, b2, w[6], 6);
  R31(b1, c1, d1, e1, a1, w[1], 15);
  R32(b2, c2, d2, e2, a2, w[9], 14);
  R31(a1, b1, c1, d1, e1, w[2], 14);
  R32(a2, b2, c2, d2, e2, w[11], 12);
  R31(e1, a1, b1, c1, d1, w[7], 8);
  R32(e2, a2, b2, c2, d2, w[8], 13);
  R31(d1, e1, a1, b1, c1, w[0], 13);
  R32(d2, e2, a2, b2, c2, w[12], 5);
  R31(c1, d1, e1, a1, b1, w[6], 6);
  R32(c2, d2, e2, a2, b2, w[2], 14);
  R31(b1, c1, d1, e1, a1, w[13], 5);
  R32(b2, c2, d2, e2, a2, w[10], 13);
  R31(a1, b1, c1, d1, e1, w[11], 12);
  R32(a2, b2, c2, d2, e2, w[0], 13);
  R31(e1, a1, b1, c1, d1, w[5], 7);
  R32(e2, a2, b2, c2, d2, w[4], 7);
  R31(d1, e1, a1, b1, c1, w[12], 5);
  R32(d2, e2, a2, b2, c2, w[13], 5);

  R41(c1, d1, e1, a1, b1, w[1], 11);
  R42(c2, d2, e2, a2, b2, w[8], 15);
  R41(b1, c1, d1, e1, a1, w[9], 12);
  R42(b2, c2, d2, e2, a2, w[6], 5);
  R41(a1, b1, c1, d1, e1, w[11], 14);
  R42(a2, b2, c2, d2, e2, w[4], 8);
  R41(e1, a1, b1, c1, d1, w[10], 15);
  R42(e2, a2, b2, c2, d2, w[1], 11);
  R41(d1, e1, a1, b1, c1, w[0], 14);
  R42(d2, e2, a2, b2, c2, w[3], 14);
  R41(c1, d1, e1, a1, b1, w[8], 15);
  R42(c2, d2, e2, a2, b2, w[11], 14);
  R41(b1, c1, d1, e1, a1, w[12], 9);
  R42(b2, c2, d2, e2, a2, w[15], 6);
  R41(a1, b1, c1, d1, e1, w[4], 8);
  R42(a2, b2, c2, d2, e2, w[0], 14);
  R41(e1, a1, b1, c1, d1, w[13], 9);
  R42(e2, a2, b2, c2, d2, w[5], 6);
  R41(d1, e1, a1, b1, c1, w[3], 14);
  R42(d2, e2, a2, b2, c2, w[12], 9);
  R41(c1, d1, e1, a1, b1, w[7], 5);
  R42(c2, d2, e2, a2, b2, w[2], 12);
  R41(b1, c1, d1, e1, a1, w[15], 6);
  R42(b2, c2, d2, e2, a2, w[13], 9);
  R41(a1, b1, c1, d1, e1, w[14], 8);
  R42(a2, b2, c2, d2, e2, w[9], 12);
  R41(e1, a1, b1, c1, d1, w[5], 6);
  R42(e2, a2, b2, c2, d2, w[7], 5);
  R41(d1, e1, a1, b1, c1, w[6], 5);
  R42(d2, e2, a2, b2, c2, w[10], 15);
  R41(c1, d1, e1, a1, b1, w[2], 12);
  R42(c2, d2, e2, a2, b2, w[14], 8);

  R51(b1, c1, d1, e1, a1, w[4], 9);
  R52(b2, c2, d2, e2, a2, w[12], 8);
  R51(a1, b1, c1, d1, e1, w[0], 15);
  R52(a2, b2, c2, d2, e2, w[15], 5);
  R51(e1, a1, b1, c1, d1, w[5], 5);
  R52(e2, a2, b2, c2, d2, w[10], 12);
  R51(d1, e1, a1, b1, c1, w[9], 11);
  R52(d2, e2, a2, b2, c2, w[4], 9);
  R51(c1, d1, e1, a1, b1, w[7], 6);
  R52(c2, d2, e2, a2, b2, w[1], 12);
  R51(b1, c1, d1, e1, a1, w[12], 8);
  R52(b2, c2, d2, e2, a2, w[5], 5);
  R51(a1, b1, c1, d1, e1, w[2], 13);
  R52(a2, b2, c2, d2, e2, w[8], 14);
  R51(e1, a1, b1, c1, d1, w[10], 12);
  R52(e2, a2, b2, c2, d2, w[7], 6);
  R51(d1, e1, a1, b1, c1, w[14], 5);
  R52(d2, e2, a2, b2, c2, w[6], 8);
  R51(c1, d1, e1, a1, b1, w[1], 12);
  R52(c2, d2, e2, a2, b2, w[2], 13);
  R51(b1, c1, d1, e1, a1, w[3], 13);
  R52(b2, c2, d2, e2, a2, w[13], 6);
  R51(a1, b1, c1, d1, e1, w[8], 14);
  R52(a2, b2, c2, d2, e2, w[14], 5);
  R51(e1, a1, b1, c1, d1, w[11], 11);
  R52(e2, a2, b2, c2, d2, w[0], 15);
  R51(d1, e1, a1, b1, c1, w[6], 8);
  R52(d2, e2, a2, b2, c2, w[3], 13);
  R51(c1, d1, e1, a1, b1, w[15], 5);
  R52(c2, d2, e2, a2, b2, w[9], 11);
  R51(b1, c1, d1, e1, a1, w[13], 6);
  R52(b2, c2, d2, e2, a2, w[11], 11);

  __m512i t = s[0];
  s[0] = add3(s[1], c1, d2);
  s[1] = add3(s[2], d1, e2);
  s[2] = add3(s[3], e1, a2);
  s[3] = add3(s[4], a1, b2);
  s[4] = add3(t, b1, c2);
}

static void ripemd160avx512_16(uint8_t* in[16], uint8_t* out[16]) {
  if (!in || !out) return;

  __m512i s[5];
  Initialize(s);

  // Safety check and padding with null pointer protection
  for (int i = 0; i < 16; ++i) {
    if (in[i]) {
      std::memcpy(in[i] + 64, pad, 56);
      std::memcpy(in[i] + 120, &sizedesc_64, 8);
    }
  }

  Transform(s, in);

  DEPACK(out[0], 15);
  DEPACK(out[1], 14);
  DEPACK(out[2], 13);
  DEPACK(out[3], 12);
  DEPACK(out[4], 11);
  DEPACK(out[5], 10);
  DEPACK(out[6], 9);
  DEPACK(out[7], 8);
  DEPACK(out[8], 7);
  DEPACK(out[9], 6);
  DEPACK(out[10], 5);
  DEPACK(out[11], 4);
  DEPACK(out[12], 3);
  DEPACK(out[13], 2);
  DEPACK(out[14], 1);
  DEPACK(out[15], 0);
}

extern "C" void ripemd160_avx512_16blocks(const uint8_t* in[16], uint8_t* out[16]) {
  if (!in || !out) return;

  uint8_t* in_mut[16];
  for (int i = 0; i < 16; ++i) in_mut[i] = const_cast<uint8_t*>(in[i]);
  ripemd160avx512_16(in_mut, out);
}

extern "C" void ripemd160_avx512_8blocks(const uint8_t* in[8], uint8_t* out[8]) {
  if (!in || !out) return;

  alignas(64) uint8_t zero[128] = {0};
  const uint8_t* in16[16];
  uint8_t* out16[16];
  for (int i = 0; i < 8; ++i) {
    in16[i] = in[i];
    out16[i] = out[i];
  }
  for (int i = 8; i < 16; ++i) {
    in16[i] = zero;
    out16[i] = zero;
  }
  ripemd160_avx512_16blocks(in16, out16);
}

extern "C" void ripemd160_avx512_1block(const uint8_t* in, uint8_t* out) {
  if (!in || !out) return;

  const uint8_t* in1[16] = {0};
  uint8_t* out1[16] = {0};
  in1[0] = in;
  out1[0] = out;
  ripemd160_avx512_16blocks(in1, out1);
}

extern "C" void ripemd160_avx512_32blocks(const uint8_t* in[32], uint8_t* out[32]) {
  if (!in || !out) return;

  alignas(64) uint8_t zero[128] = {0};
  const uint8_t* in16a[16];
  const uint8_t* in16b[16];
  uint8_t* out16a[16];
  uint8_t* out16b[16];

  // first 16 blocks
  for (int i = 0; i < 16; ++i) {
    in16a[i] = in[i];
    out16a[i] = out[i];
  }

  // second 16 blocks
  for (int i = 0; i < 16; ++i) {
    in16b[i] = in[i + 16];
    out16b[i] = out[i + 16];
  }

  // fallback zero-fill if any block is nullptr
  for (int i = 0; i < 16; ++i) {
    if (!in16a[i]) in16a[i] = zero;
    if (!out16a[i]) out16a[i] = zero;
    if (!in16b[i]) in16b[i] = zero;
    if (!out16b[i]) out16b[i] = zero;
  }

  ripemd160_avx512_16blocks(in16a, out16a);
  ripemd160_avx512_16blocks(in16b, out16b);
}
