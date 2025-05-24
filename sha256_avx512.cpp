#include "sha256_avx512.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

namespace _sha256avx512 {

#define ALIGN64 __attribute__((aligned(64)))

// Initial hash values for 32 parallel lanes (explicitly unrolled)
static const ALIGN64 uint32_t initConstants[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// SHA-256 round constants
static const ALIGN64 uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define SHR(x, n) _mm512_srli_epi32(x, n)
#define ROTR(x, n) _mm512_or_si512(SHR(x, n), _mm512_slli_epi32(x, 32 - n))
#define CH(x, y, z) _mm512_xor_si512(_mm512_and_si512(x, y), _mm512_andnot_si512(x, z))
#define MAJ(x, y, z) \
  _mm512_or_si512(_mm512_and_si512(x, y), _mm512_and_si512(z, _mm512_or_si512(x, y)))

#define EP0(x) (_mm512_xor_si512(_mm512_xor_si512(ROTR(x, 2), ROTR(x, 13)), ROTR(x, 22)))
#define EP1(x) (_mm512_xor_si512(_mm512_xor_si512(ROTR(x, 6), ROTR(x, 11)), ROTR(x, 25)))
#define SIG0(x) (_mm512_xor_si512(_mm512_xor_si512(ROTR(x, 7), ROTR(x, 18)), SHR(x, 3)))
#define SIG1(x) (_mm512_xor_si512(_mm512_xor_si512(ROTR(x, 17), ROTR(x, 19)), SHR(x, 10)))

void Initialize(__m512i state[8]) {
    // Każdy lane w rejestrze to inny blok - ustawiamy na 32x taki sam init
    for (int i = 0; i < 8; i++) {
        state[i] = _mm512_set1_epi32(initConstants[i]);
    }
}

void Transform(__m512i state[8], const uint8_t* data[32]) {
    __m512i a = state[0], b = state[1], c = state[2], d = state[3];
    __m512i e = state[4], f = state[5], g = state[6], h = state[7];
    __m512i W[64];
    __m512i T1, T2;

    // Message schedule
    for (int t = 0; t < 16; t++) {
        ALIGN64 uint32_t wt[16][32];
        for (int lane = 0; lane < 32; lane++) {
            const uint8_t* ptr = data[lane] + t * 4;
            wt[0][lane] = (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
        }
        W[t] = _mm512_load_epi32(wt[0]);
    }
    for (int t = 16; t < 64; t++) {
        W[t] = _mm512_add_epi32(_mm512_add_epi32(SIG1(W[t - 2]), W[t - 7]),
                                _mm512_add_epi32(SIG0(W[t - 15]), W[t - 16]));
    }

    // Main compression loop
    for (int t = 0; t < 64; t++) {
        T1 = _mm512_add_epi32(
            _mm512_add_epi32(_mm512_add_epi32(h, EP1(e)),
                _mm512_add_epi32(CH(e, f, g), _mm512_set1_epi32(K[t]))),
            W[t]);
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

    // Update state
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

    // Store results
    for (int lane = 0; lane < 32; lane++) {
        alignas(64) uint32_t tmp[16];
        for (int i = 0; i < 8; i++) {
            _mm512_store_si512(tmp, state[i]);
            uint32_t val = tmp[lane];
#if defined(_MSC_VER)
            val = _byteswap_ulong(val);
#else
            val = __builtin_bswap32(val);
#endif
            memcpy(out[lane] + i * 4, &val, 4);
        }
    }
}

// Uniwersalne batchowanie dla mniejszych batchy przez batch 32x

extern "C" void sha256_avx512_16blocks(const uint8_t* in[16], uint8_t* out[16]) {
    alignas(64) uint8_t zero[64] = {0};
    const uint8_t* data[32];
    uint8_t*      hashes[32];
    for (int i = 0; i < 16; ++i) { data[i] = in[i]; hashes[i] = out[i]; }
    for (int i = 16; i < 32; ++i) { data[i] = zero; hashes[i] = zero; }
    sha256_avx512_32blocks(data, hashes);
}
extern "C" void sha256_avx512_8blocks(const uint8_t* in[8], uint8_t* out[8]) {
    alignas(64) uint8_t zero[64] = {0};
    const uint8_t* data[32];
    uint8_t*      hashes[32];
    for (int i = 0; i < 8; ++i) { data[i] = in[i]; hashes[i] = out[i]; }
    for (int i = 8; i < 32; ++i) { data[i] = zero; hashes[i] = zero; }
    sha256_avx512_32blocks(data, hashes);
}
extern "C" void sha256_avx512_4blocks(const uint8_t* in[4], uint8_t* out[4]) {
    alignas(64) uint8_t zero[64] = {0};
    const uint8_t* data[32];
    uint8_t*      hashes[32];
    for (int i = 0; i < 4; ++i) { data[i] = in[i]; hashes[i] = out[i]; }
    for (int i = 4; i < 32; ++i) { data[i] = zero; hashes[i] = zero; }
    sha256_avx512_32blocks(data, hashes);
}
extern "C" void sha256_avx512_1block(const uint8_t* in, uint8_t* out) {
    alignas(64) uint8_t zero[64] = {0};
    const uint8_t* data[32];
    uint8_t*      hashes[32];
    data[0] = in; hashes[0] = out;
    for (int i = 1; i < 32; ++i) { data[i] = zero; hashes[i] = zero; }
    sha256_avx512_32blocks(data, hashes);
}
