#ifndef SHA256_AVX512_H
#define SHA256_AVX512_H

#include <immintrin.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Process 32 blocks in parallel
void sha256_avx512_32blocks(const uint8_t* data[32], uint8_t* out_hashes[32]);

#ifdef __cplusplus
}
#endif

#endif // SHA256_AVX512_H