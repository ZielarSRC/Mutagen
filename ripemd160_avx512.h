#ifndef RIPEMD160_AVX512_H
#define RIPEMD160_AVX512_H

#include <immintrin.h>

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Full batch AVX-512: 16 blocks with 64 bits, 16 hashes with 20 bits
void ripemd160_avx512_16blocks(const uint8_t* in[16], uint8_t* out[16]);

// Batch 8x
void ripemd160_avx512_8blocks(const uint8_t* in[8], uint8_t* out[8]);

// Batch 1x
void ripemd160_avx512_1block(const uint8_t* in, uint8_t* out);

#ifdef __cplusplus
}
#endif

#endif  // RIPEMD160_AVX512_H
