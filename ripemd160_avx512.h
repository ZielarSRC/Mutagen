#ifndef RIPEMD160_AVX512_H
#define RIPEMD160_AVX512_H

#include <immintrin.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Pełny batch AVX-512: 16 bloków wejściowych po 64 bajty, 16 hashy po 20 bajtów
void ripemd160_avx512_16blocks(const uint8_t* in[16], uint8_t* out[16]);

// Batch 8x (wywołuje 16x, resztę zeruje)
void ripemd160_avx512_8blocks(const uint8_t* in[8], uint8_t* out[8]);

// Batch 1x (wywołuje 16x, resztę zeruje)
void ripemd160_avx512_1block(const uint8_t* in, uint8_t* out);

#ifdef __cplusplus
}
#endif

#endif // RIPEMD160_AVX512_H
