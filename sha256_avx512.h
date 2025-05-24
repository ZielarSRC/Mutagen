#ifndef SHA256_AVX512_H
#define SHA256_AVX512_H

#include <immintrin.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum performance AVX-512: batch 32 blocks (64B for block, 32x32B hash)
void sha256_avx512_32blocks(const uint8_t* in[32], uint8_t* out[32]);

// Batch 16x
void sha256_avx512_16blocks(const uint8_t* in[16], uint8_t* out[16]);

// Batch 8x
void sha256_avx512_8blocks(const uint8_t* in[8], uint8_t* out[8]);

// Batch 4x
void sha256_avx512_4blocks(const uint8_t* in[4], uint8_t* out[4]);

// Batch 1x
void sha256_avx512_1block(const uint8_t* in, uint8_t* out);

#ifdef __cplusplus
}
#endif

#endif // SHA256_AVX512_H
