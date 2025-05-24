#include "Int.h"
#include "IntGroup.h"
#include <string.h>
#include <math.h>
#include <emmintrin.h>
#include <stdexcept>
#include <sstream>
#include <cstdio>
#include <cctype>
#include <iomanip>

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

Int _ONE((uint64_t)1);
Int Int::P;

// Forward declarations for assembly functions
extern "C" void shiftL(unsigned char n, uint64_t* d);
extern "C" void shiftR(unsigned char n, uint64_t* d);
extern "C" void imm_mul(uint64_t* a, uint64_t b, uint64_t* r, uint64_t* carry);
extern "C" void imm_imul(uint64_t* a, int64_t b, uint64_t* r, uint64_t* carry);

// Inline assembly function placeholders if not available
#ifndef WIN64
inline void shiftL(unsigned char n, uint64_t* d) {
    if (n == 0) return;
    uint64_t carry = 0;
    for (int i = 0; i < NB64BLOCK; i++) {
        uint64_t newCarry = d[i] >> (64 - n);
        d[i] = (d[i] << n) | carry;
        carry = newCarry;
    }
}

inline void shiftR(unsigned char n, uint64_t* d) {
    if (n == 0) return;
    uint64_t carry = 0;
    for (int i = NB64BLOCK - 1; i >= 0; i--) {
        uint64_t newCarry = d[i] << (64 - n);
        d[i] = (d[i] >> n) | carry;
        carry = newCarry;
    }
}

inline void imm_mul(uint64_t* a, uint64_t b, uint64_t* r, uint64_t* carry) {
    *carry = 0;
    for (int i = 0; i < NB64BLOCK; i++) {
        __uint128_t prod = (__uint128_t)a[i] * b + *carry;
        r[i] = (uint64_t)prod;
        *carry = prod >> 64;
    }
}

inline void imm_imul(uint64_t* a, int64_t b, uint64_t* r, uint64_t* carry) {
    if (b < 0) {
        // Negate a and make b positive
        for (int i = 0; i < NB64BLOCK; i++) r[i] = ~a[i];
        // Add 1 to complete 2's complement
        unsigned char c = 1;
        for (int i = 0; i < NB64BLOCK; i++) {
            c = _addcarry_u64(c, r[i], 0, r + i);
            if (!c) break;
        }
        b = -b;
        imm_mul(r, (uint64_t)b, r, carry);
    } else {
        imm_mul(a, (uint64_t)b, r, carry);
    }
}

inline bool isStrictGreater128(uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl) {
    return (ah > bh) || (ah == bh && al > bl);
}
#endif

// -------------------
// Constructors
// -------------------
Int::Int() { CLEAR(); }

Int::Int(int64_t i64) {
    if (i64 < 0) {
        CLEARFF();
    } else {
        CLEAR();
    }
    bits64[0] = i64;
}

Int::Int(uint64_t u64) {
    CLEAR();
    bits64[0] = u64;
}

Int::Int(const Int *a) {
    if(a) Set(a);
    else CLEAR();
}

// -------------------
void Int::CLEAR() {
    memset(bits64, 0, NB64BLOCK*8);
}

void Int::CLEARFF() {
    memset(bits64, 0xFF, NB64BLOCK * 8);
}

// -------------------
void Int::Set(const Int *a) {
    for (int i = 0; i < NB64BLOCK; i++)
        bits64[i] = a->bits64[i];
}

// -------------------
// AVX-512 optimized XOR
void Int::Xor(const Int *a) {
    if (!a) return;

#if defined(__AVX512F__)
    // Use AVX-512 for optimal performance on Xeon Platinum
    __m512i *this_vec = (__m512i*)bits64;
    const __m512i *a_vec = (const __m512i*)a->bits64;
    
    for (int i = 0; i < (NB64BLOCK * 8) / 64; i++) {
        __m512i x = _mm512_loadu_si512(&a_vec[i]);
        __m512i y = _mm512_loadu_si512(&this_vec[i]);
        __m512i z = _mm512_xor_si512(x, y);
        _mm512_storeu_si512(&this_vec[i], z);
    }
    
    // Handle remaining bytes
    int remaining = (NB64BLOCK * 8) % 64;
    if (remaining > 0) {
        uint8_t *this_bytes = (uint8_t*)bits64;
        const uint8_t *a_bytes = (const uint8_t*)a->bits64;
        int start = ((NB64BLOCK * 8) / 64) * 64;
        for (int i = 0; i < remaining; i++) {
            this_bytes[start + i] ^= a_bytes[start + i];
        }
    }
#else
    // Fallback for systems without AVX-512
    for (int i = 0; i < NB64BLOCK; i++) {
        bits64[i] ^= a->bits64[i];
    }
#endif
}

// -------------------
// Addition operations
void Int::Add(uint64_t a) {
    unsigned char c = 0;
    c = _addcarry_u64(c, bits64[0], a, bits64 + 0);
    c = _addcarry_u64(c, bits64[1], 0, bits64 + 1);
    c = _addcarry_u64(c, bits64[2], 0, bits64 + 2);
    c = _addcarry_u64(c, bits64[3], 0, bits64 + 3);
    c = _addcarry_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
    c = _addcarry_u64(c, bits64[5], 0, bits64 + 5);
    c = _addcarry_u64(c, bits64[6], 0, bits64 + 6);
    c = _addcarry_u64(c, bits64[7], 0, bits64 + 7);
    c = _addcarry_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

void Int::Add(const Int *a) {
    unsigned char c = 0;
    c = _addcarry_u64(c, bits64[0], a->bits64[0], bits64 + 0);
    c = _addcarry_u64(c, bits64[1], a->bits64[1], bits64 + 1);
    c = _addcarry_u64(c, bits64[2], a->bits64[2], bits64 + 2);
    c = _addcarry_u64(c, bits64[3], a->bits64[3], bits64 + 3);
    c = _addcarry_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
    c = _addcarry_u64(c, bits64[5], a->bits64[5], bits64 + 5);
    c = _addcarry_u64(c, bits64[6], a->bits64[6], bits64 + 6);
    c = _addcarry_u64(c, bits64[7], a->bits64[7], bits64 + 7);
    c = _addcarry_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
}

void Int::Add(const Int *a, const Int *b) {
    unsigned char c = 0;
    c = _addcarry_u64(c, a->bits64[0], b->bits64[0], bits64 + 0);
    c = _addcarry_u64(c, a->bits64[1], b->bits64[1], bits64 + 1);
    c = _addcarry_u64(c, a->bits64[2], b->bits64[2], bits64 + 2);
    c = _addcarry_u64(c, a->bits64[3], b->bits64[3], bits64 + 3);
    c = _addcarry_u64(c, a->bits64[4], b->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
    c = _addcarry_u64(c, a->bits64[5], b->bits64[5], bits64 + 5);
    c = _addcarry_u64(c, a->bits64[6], b->bits64[6], bits64 + 6);
    c = _addcarry_u64(c, a->bits64[7], b->bits64[7], bits64 + 7);
    c = _addcarry_u64(c, a->bits64[8], b->bits64[8], bits64 + 8);
#endif
}

void Int::AddOne() {
    unsigned char c = 0;
    c = _addcarry_u64(c, bits64[0], 1, bits64 + 0);
    c = _addcarry_u64(c, bits64[1], 0, bits64 + 1);
    c = _addcarry_u64(c, bits64[2], 0, bits64 + 2);
    c = _addcarry_u64(c, bits64[3], 0, bits64 + 3);
    c = _addcarry_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
    c = _addcarry_u64(c, bits64[5], 0, bits64 + 5);
    c = _addcarry_u64(c, bits64[6], 0, bits64 + 6);
    c = _addcarry_u64(c, bits64[7], 0, bits64 + 7);
    c = _addcarry_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

// Helper functions for matrix operations
uint64_t Int::AddCh(const Int *a, uint64_t ca, const Int *b, uint64_t cb) {
    uint64_t carry;
    unsigned char c = 0;
    c = _addcarry_u64(c, a->bits64[0], b->bits64[0], bits64 + 0);
    c = _addcarry_u64(c, a->bits64[1], b->bits64[1], bits64 + 1);
    c = _addcarry_u64(c, a->bits64[2], b->bits64[2], bits64 + 2);
    c = _addcarry_u64(c, a->bits64[3], b->bits64[3], bits64 + 3);
    c = _addcarry_u64(c, a->bits64[4], b->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
    c = _addcarry_u64(c, a->bits64[5], b->bits64[5], bits64 + 5);
    c = _addcarry_u64(c, a->bits64[6], b->bits64[6], bits64 + 6);
    c = _addcarry_u64(c, a->bits64[7], b->bits64[7], bits64 + 7);
    c = _addcarry_u64(c, a->bits64[8], b->bits64[8], bits64 + 8);
#endif
    _addcarry_u64(c, ca, cb, &carry);
    return carry;
}

uint64_t Int::AddCh(const Int *a, uint64_t ca) {
    uint64_t carry;
    unsigned char c = 0;
    c = _addcarry_u64(c, bits64[0], a->bits64[0], bits64 + 0);
    c = _addcarry_u64(c, bits64[1], a->bits64[1], bits64 + 1);
    c = _addcarry_u64(c, bits64[2], a->bits64[2], bits64 + 2);
    c = _addcarry_u64(c, bits64[3], a->bits64[3], bits64 + 3);
    c = _addcarry_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
    c = _addcarry_u64(c, bits64[5], a->bits64[5], bits64 + 5);
    c = _addcarry_u64(c, bits64[6], a->bits64[6], bits64 + 6);
    c = _addcarry_u64(c, bits64[7], a->bits64[7], bits64 + 7);
    c = _addcarry_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
    _addcarry_u64(c, ca, 0, &carry);
    return carry;
}

uint64_t Int::AddC(const Int *a) {
    unsigned char c = 0;
    c = _addcarry_u64(c, bits64[0], a->bits64[0], bits64 + 0);
    c = _addcarry_u64(c, bits64[1], a->bits64[1], bits64 + 1);
    c = _addcarry_u64(c, bits64[2], a->bits64[2], bits64 + 2);
    c = _addcarry_u64(c, bits64[3], a->bits64[3], bits64 + 3);
    c = _addcarry_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
    c = _addcarry_u64(c, bits64[5], a->bits64[5], bits64 + 5);
    c = _addcarry_u64(c, bits64[6], a->bits64[6], bits64 + 6);
    c = _addcarry_u64(c, bits64[7], a->bits64[7], bits64 + 7);
    c = _addcarry_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
    return c;
}

void Int::AddAndShift(const Int *a, const Int *b, uint64_t cH) {
    unsigned char c = 0;
    c = _addcarry_u64(c, b->bits64[0], a->bits64[0], bits64 + 0);
    c = _addcarry_u64(c, b->bits64[1], a->bits64[1], bits64 + 0);
    c = _addcarry_u64(c, b->bits64[2], a->bits64[2], bits64 + 1);
    c = _addcarry_u64(c, b->bits64[3], a->bits64[3], bits64 + 2);
    c = _addcarry_u64(c, b->bits64[4], a->bits64[4], bits64 + 3);
#if NB64BLOCK > 5
    c = _addcarry_u64(c, b->bits64[5], a->bits64[5], bits64 + 4);
    c = _addcarry_u64(c, b->bits64[6], a->bits64[6], bits64 + 5);
    c = _addcarry_u64(c, b->bits64[7], a->bits64[7], bits64 + 6);
    c = _addcarry_u64(c, b->bits64[8], a->bits64[8], bits64 + 7);
#endif
    bits64[NB64BLOCK - 1] = c + cH;
}

// -------------------
// Subtraction operations
void Int::Sub(uint64_t a) {
    unsigned char c = 0;
    c = _subborrow_u64(c, bits64[0], a, bits64 + 0);
    c = _subborrow_u64(c, bits64[1], 0, bits64 + 1);
    c = _subborrow_u64(c, bits64[2], 0, bits64 + 2);
    c = _subborrow_u64(c, bits64[3], 0, bits64 + 3);
    c = _subborrow_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
    c = _subborrow_u64(c, bits64[5], 0, bits64 + 5);
    c = _subborrow_u64(c, bits64[6], 0, bits64 + 6);
    c = _subborrow_u64(c, bits64[7], 0, bits64 + 7);
    c = _subborrow_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

void Int::Sub(const Int *a) {
    unsigned char c = 0;
    c = _subborrow_u64(c, bits64[0], a->bits64[0], bits64 + 0);
    c = _subborrow_u64(c, bits64[1], a->bits64[1], bits64 + 1);
    c = _subborrow_u64(c, bits64[2], a->bits64[2], bits64 + 2);
    c = _subborrow_u64(c, bits64[3], a->bits64[3], bits64 + 3);
    c = _subborrow_u64(c, bits64[4], a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
    c = _subborrow_u64(c, bits64[5], a->bits64[5], bits64 + 5);
    c = _subborrow_u64(c, bits64[6], a->bits64[6], bits64 + 6);
    c = _subborrow_u64(c, bits64[7], a->bits64[7], bits64 + 7);
    c = _subborrow_u64(c, bits64[8], a->bits64[8], bits64 + 8);
#endif
}

void Int::Sub(const Int *a, const Int *b) {
    unsigned char c = 0;
    c = _subborrow_u64(c, a->bits64[0], b->bits64[0], bits64 + 0);
    c = _subborrow_u64(c, a->bits64[1], b->bits64[1], bits64 + 1);
    c = _subborrow_u64(c, a->bits64[2], b->bits64[2], bits64 + 2);
    c = _subborrow_u64(c, a->bits64[3], b->bits64[3], bits64 + 3);
    c = _subborrow_u64(c, a->bits64[4], b->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
    c = _subborrow_u64(c, a->bits64[5], b->bits64[5], bits64 + 5);
    c = _subborrow_u64(c, a->bits64[6], b->bits64[6], bits64 + 6);
    c = _subborrow_u64(c, a->bits64[7], b->bits64[7], bits64 + 7);
    c = _subborrow_u64(c, a->bits64[8], b->bits64[8], bits64 + 8);
#endif
}

void Int::SubOne() {
    unsigned char c = 0;
    c = _subborrow_u64(c, bits64[0], 1, bits64 + 0);
    c = _subborrow_u64(c, bits64[1], 0, bits64 + 1);
    c = _subborrow_u64(c, bits64[2], 0, bits64 + 2);
    c = _subborrow_u64(c, bits64[3], 0, bits64 + 3);
    c = _subborrow_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
    c = _subborrow_u64(c, bits64[5], 0, bits64 + 5);
    c = _subborrow_u64(c, bits64[6], 0, bits64 + 6);
    c = _subborrow_u64(c, bits64[7], 0, bits64 + 7);
    c = _subborrow_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

void Int::Neg() {
    unsigned char c = 0;
    c = _subborrow_u64(c, 0, bits64[0], bits64 + 0);
    c = _subborrow_u64(c, 0, bits64[1], bits64 + 1);
    c = _subborrow_u64(c, 0, bits64[2], bits64 + 2);
    c = _subborrow_u64(c, 0, bits64[3], bits64 + 3);
    c = _subborrow_u64(c, 0, bits64[4], bits64 + 4);
#if NB64BLOCK > 5
    c = _subborrow_u64(c, 0, bits64[5], bits64 + 5);
    c = _subborrow_u64(c, 0, bits64[6], bits64 + 6);
    c = _subborrow_u64(c, 0, bits64[7], bits64 + 7);
    c = _subborrow_u64(c, 0, bits64[8], bits64 + 8);
#endif
}

// -------------------
// Comparison operations
bool Int::IsGreater(const Int *a) const {
    int i = NB64BLOCK-1;
    for(; i >= 0; --i) {
        if(a->bits64[i] != bits64[i])
            break;
    }
    if(i >= 0) {
        return bits64[i] > a->bits64[i];
    } else {
        return false;
    }
}

bool Int::IsLower(const Int *a) const {
    int i = NB64BLOCK-1;
    for(; i >= 0; --i) {
        if(a->bits64[i] != bits64[i])
            break;
    }
    if(i >= 0) {
        return bits64[i] < a->bits64[i];
    } else {
        return false;
    }
}

bool Int::IsGreaterOrEqual(const Int *a) const {
    Int p;
    p.Sub(this, a);
    return p.IsPositive();
}

bool Int::IsLowerOrEqual(const Int *a) const {
    int i = NB64BLOCK-1;
    for(; i >= 0; --i) {
        if(a->bits64[i] != bits64[i])
            break;
    }
    if(i >= 0) {
        return bits64[i] < a->bits64[i];
    } else {
        return true;
    }
}

bool Int::IsEqual(const Int *a) const {
    return
#if NB64BLOCK > 5
           (bits64[8] == a->bits64[8]) &&
           (bits64[7] == a->bits64[7]) &&
           (bits64[6] == a->bits64[6]) &&
           (bits64[5] == a->bits64[5]) &&
#endif
           (bits64[4] == a->bits64[4]) &&
           (bits64[3] == a->bits64[3]) &&
           (bits64[2] == a->bits64[2]) &&
           (bits64[1] == a->bits64[1]) &&
           (bits64[0] == a->bits64[0]);
}

bool Int::IsOne() const {
    return IsEqual(&_ONE);
}

bool Int::IsZero() const {
#if NB64BLOCK > 5
    return (bits64[8] | bits64[7] | bits64[6] | bits64[5] | bits64[4] | bits64[3] | bits64[2] | bits64[1] | bits64[0]) == 0;
#else
    return (bits64[4] | bits64[3] | bits64[2] | bits64[1] | bits64[0]) == 0;
#endif
}

bool Int::IsStrictPositive() const { return IsPositive() && !IsZero(); }
bool Int::IsPositive() const { return (int64_t)(bits64[NB64BLOCK - 1]) >= 0; }
bool Int::IsNegative() const { return (int64_t)(bits64[NB64BLOCK - 1]) < 0; }
bool Int::IsEven() const { return (bits[0] & 0x1) == 0; }
bool Int::IsOdd() const { return (bits[0] & 0x1) == 1; }

// -------------------
// Bit manipulation
void Int::SetInt32(uint32_t value) {
    CLEAR();
    bits[0] = value;
}

uint32_t Int::GetInt32() const {
    return bits[0];
}

unsigned char Int::GetByte(int n) const {
    const unsigned char *bbPtr = (const unsigned char *)bits;
    return bbPtr[n];
}

void Int::Set32Bytes(const uint8_t *bytes) {
    CLEAR();
    const uint64_t *ptr = (const uint64_t *)bytes;
    bits64[3] = _byteswap_uint64(ptr[0]);
    bits64[2] = _byteswap_uint64(ptr[1]);
    bits64[1] = _byteswap_uint64(ptr[2]);
    bits64[0] = _byteswap_uint64(ptr[3]);
}

void Int::Get32Bytes(uint8_t *buff) const {
    uint64_t *ptr = (uint64_t *)buff;
    ptr[3] = _byteswap_uint64(bits64[0]);
    ptr[2] = _byteswap_uint64(bits64[1]);
    ptr[1] = _byteswap_uint64(bits64[2]);
    ptr[0] = _byteswap_uint64(bits64[3]);
}

void Int::SetByte(int n, unsigned char byte) {
    unsigned char *bbPtr = (unsigned char *)bits;
    bbPtr[n] = byte;
}

void Int::SetDWord(int n, uint32_t b) {
    bits[n] = b;
}

void Int::SetQWord(int n, uint64_t b) {
    bits64[n] = b;
}

int Int::GetBit(uint32_t n) const {
    uint32_t nb64 = n / 64;
    uint32_t nb = n % 64;
    return (bits64[nb64] >> nb) & 1;
}

// -------------------
// Shift operations
void Int::ShiftL32Bit() {
    for(int i = NB32BLOCK-1; i > 0; i--) {
        bits[i] = bits[i-1];
    }
    bits[0] = 0;
}

void Int::ShiftL64Bit() {
    for (int i = NB64BLOCK-1; i > 0; i--) {
        bits64[i] = bits64[i - 1];
    }
    bits64[0] = 0;
}

void Int::ShiftL64BitAndSub(const Int *a, int n) {
    Int b;
    int i = NB64BLOCK-1;

    for(; i >= n; i--)
        b.bits64[i] = ~a->bits64[i-n];
    for(; i >= 0; i--)
        b.bits64[i] = 0xFFFFFFFFFFFFFFFFULL;

    Add(&b);
    AddOne();
}

void Int::ShiftL(uint32_t n) {
    if(n == 0) return;
    if(n < 64) {
        shiftL((unsigned char)n, bits64);
    } else {
        uint32_t nb64 = n/64;
        uint32_t nb = n%64;
        for(uint32_t i = 0; i < nb64; i++) ShiftL64Bit();
        shiftL((unsigned char)nb, bits64);
    }
}

void Int::ShiftR32Bit() {
    for(int i = 0; i < NB32BLOCK-1; i++) {
        bits[i] = bits[i+1];
    }
    if(((int32_t)bits[NB32BLOCK-2]) < 0)
        bits[NB32BLOCK-1] = 0xFFFFFFFF;
    else
        bits[NB32BLOCK-1] = 0;
}

void Int::ShiftR64Bit() {
    for (int i = 0; i < NB64BLOCK - 1; i++) {
        bits64[i] = bits64[i + 1];
    }
    if (((int64_t)bits64[NB64BLOCK - 2]) < 0)
        bits64[NB64BLOCK - 1] = 0xFFFFFFFFFFFFFFFFULL;
    else
        bits64[NB64BLOCK - 1] = 0;
}

void Int::ShiftR(uint32_t n) {
    if(n == 0) return;
    if(n < 64) {
        shiftR((unsigned char)n, bits64);
    } else {
        uint32_t nb64 = n/64;
        uint32_t nb = n%64;
        for(uint32_t i = 0; i < nb64; i++) ShiftR64Bit();
        shiftR((unsigned char)nb, bits64);
    }
}

void Int::SwapBit(int bitNumber) {
    uint32_t nb64 = bitNumber / 64;
    uint32_t nb = bitNumber % 64;
    uint64_t mask = 1ULL << nb;
    if(bits64[nb64] & mask) {
        bits64[nb64] &= ~mask;
    } else {
        bits64[nb64] |= mask;
    }
}

// -------------------
// Multiplication operations
void Int::Mult(const Int *a) {
    Int b(this);
    Mult(a, &b);
}

uint64_t Int::IMult(int64_t a) {
    uint64_t carry;
    if (a < 0LL) {
        a = -a;
        Neg();
    }
    imm_imul(bits64, a, bits64, &carry);
    return carry;
}

uint64_t Int::Mult(uint64_t a) {
    uint64_t carry;
    imm_mul(bits64, a, bits64, &carry);
    return carry;
}

uint64_t Int::IMult(const Int *a, int64_t b) {
    uint64_t carry;
    if (b < 0LL) {
        unsigned char c = 0;
        c = _subborrow_u64(c, 0, a->bits64[0], bits64 + 0);
        c = _subborrow_u64(c, 0, a->bits64[1], bits64 + 1);
        c = _subborrow_u64(c, 0, a->bits64[2], bits64 + 2);
        c = _subborrow_u64(c, 0, a->bits64[3], bits64 + 3);
        c = _subborrow_u64(c, 0, a->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
        c = _subborrow_u64(c, 0, a->bits64[5], bits64 + 5);
        c = _subborrow_u64(c, 0, a->bits64[6], bits64 + 6);
        c = _subborrow_u64(c, 0, a->bits64[7], bits64 + 7);
        c = _subborrow_u64(c, 0, a->bits64[8], bits64 + 8);
#endif
        b = -b;
    } else {
        Set(a);
    }
    imm_imul(bits64, b, bits64, &carry);
    return carry;
}

uint64_t Int::Mult(const Int *a, uint64_t b) {
    uint64_t carry;
    imm_mul(const_cast<uint64_t*>(a->bits64), b, bits64, &carry);
    return carry;
}

void Int::Mult(const Int *a, const Int *b) {
    unsigned char c = 0;
    uint64_t h;
    uint64_t pr = 0;
    uint64_t carryh = 0;
    uint64_t carryl = 0;

    bits64[0] = _umul128(a->bits64[0], b->bits64[0], &pr);

    for (int i = 1; i < NB64BLOCK; i++) {
        for (int j = 0; j <= i; j++) {
            c = _addcarry_u64(c, _umul128(a->bits64[j], b->bits64[i - j], &h), pr, &pr);
            c = _addcarry_u64(c, carryl, h, &carryl);
            c = _addcarry_u64(c, carryh, 0, &carryh);
        }
        bits64[i] = pr;
        pr = carryl;
        carryl = carryh;
        carryh = 0;
    }
}

// -------------------
// Matrix operations
void Int::MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22, uint64_t *cu, uint64_t *cv) {
    Int t1, t2, t3, t4;
    uint64_t c1, c2, c3, c4;
    c1 = t1.IMult(u, _11);
    c2 = t2.IMult(v, _12);
    c3 = t3.IMult(u, _21);
    c4 = t4.IMult(v, _22);
    *cu = u->AddCh(&t1, c1, &t2, c2);
    *cv = v->AddCh(&t3, c3, &t4, c4);
}

void Int::MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22) {
    Int t1, t2, t3, t4;
    t1.IMult(u, _11);
    t2.IMult(v, _12);
    t3.IMult(u, _21);
    t4.IMult(v, _22);
    u->Add(&t1, &t2);
    v->Add(&t3, &t4);
}

// -------------------
// Utility functions
void Int::Abs() {
    if (IsNegative())
        Neg();
}

double Int::ToDouble() const {
    double base = 1.0;
    double sum = 0;
    double pw32 = pow(2.0, 32.0);
    for(int i = 0; i < NB32BLOCK; i++) {
        sum += (double)(bits[i]) * base;
        base *= pw32;
    }
    return sum;
}

int Int::GetSize() const {
    int i = NB32BLOCK-1;
    while(i > 0 && bits[i] == 0) i--;
    return i+1;
}

int Int::GetSize64() const {
    int i = NB64BLOCK - 1;
    while(i > 0 && bits64[i] == 0) i--;
    return i + 1;
}

int Int::GetBitLength() const {
    Int t(this);
    if(IsNegative())
        t.Neg();
    int i = NB64BLOCK-1;
    while(i >= 0 && t.bits64[i] == 0) i--;
    if(i < 0) return 0;
    return (int)((64-LZC(t.bits64[i])) + i*64);
}

int Int::GetLowestBit() const {
    // Assume this!=0
    int b = 0;
    while(GetBit(b) == 0) b++;
    return b;
}

void Int::MaskByte(int n) {
    for (int i = n; i < NB32BLOCK; i++)
        bits[i] = 0;
}

// -------------------
// String operations
void Int::SetBase10(const char *value) {
    CLEAR();
    Int pw((uint64_t)1);
    Int c;
    int lgth = (int)strlen(value);
    for(int i = lgth-1; i >= 0; i--) {
        uint32_t id = (uint32_t)(value[i]-'0');
        c.Set(&pw);
        c.Mult(id);
        Add(&c);
        pw.Mult(10);
    }
}

void Int::SetBase16(const char *value) {
    CLEAR();
    Int pw((uint64_t)1);
    Int c;
    int lgth = (int)strlen(value);
    for(int i = lgth-1; i >= 0; i--) {
        char ch = toupper(value[i]);
        uint32_t id;
        if(ch >= '0' && ch <= '9') {
            id = ch - '0';
        } else if(ch >= 'A' && ch <= 'F') {
            id = ch - 'A' + 10;
        } else throw std::runtime_error("Invalid hex character");
        
        c.Set(&pw);
        c.Mult(id);
        Add(&c);
        pw.Mult(16);
    }
}

std::string Int::GetBase10() const {
    return GetBaseN(10, "0123456789");
}

std::string Int::GetBase16() const {
    return GetBaseN(16, "0123456789ABCDEF");
}

std::string Int::GetBlockStr() const {
    std::stringstream ss;
    for (int i = NB32BLOCK-3; i >= 0; i--) {
        if(i != NB32BLOCK-3) ss << " ";
        ss << std::hex << std::setfill('0') << std::setw(8) << std::uppercase << bits[i];
    }
    return ss.str();
}

std::string Int::GetC64Str(int nbDigit) const {
    std::stringstream ss;
    ss << "{";
    for (int i = 0; i < nbDigit; i++) {
        if (i > 0) ss << ",";
        if (bits64[i] != 0) {
            ss << "0x" << std::hex << bits64[i] << "ULL";
        } else {
            ss << "0ULL";
        }
    }
    ss << "}";
    return ss.str();
}

void Int::SetBaseN(int n, const char *charset, const char *value) {
    CLEAR();
    Int pw((uint64_t)1);
    Int nb((uint64_t)n);
    Int c;

    int lgth = (int)strlen(value);
    for(int i = lgth-1; i >= 0; i--) {
        const char *p = strchr(charset, toupper(value[i]));
        if(!p) {
            printf("Invalid charset !!\n");
            return;
        }
        int id = (int)(p-charset);
        c.SetInt32(id);
        c.Mult(&pw);
        Add(&c);
        pw.Mult(&nb);
    }
}

std::string Int::GetBaseN(int n, const char *charset) const {
    std::string ret;
    Int N(this);
    int isNegative = N.IsNegative();
    if (isNegative) N.Neg();

    unsigned char digits[1024];
    memset(digits, 0, sizeof(digits));

    int digitslen = 1;
    for (int i = 0; i < NB64BLOCK * 8; i++) {
        unsigned int carry = N.GetByte(NB64BLOCK*8 - i - 1);
        for (int j = 0; j < digitslen; j++) {
            carry += (unsigned int)(digits[j]) << 8;
            digits[j] = (unsigned char)(carry % n);
            carry /= n;
        }
        while (carry > 0) {
            digits[digitslen++] = (unsigned char)(carry % n);
            carry /= n;
        }
    }

    if (isNegative)
        ret.push_back('-');

    for (int i = 0; i < digitslen; i++)
        ret.push_back(charset[digits[digitslen - 1 - i]]);

    if (ret.length() == 0)
        ret.push_back('0');

    return ret;
}

// -------------------
// Division operations
void Int::MultModN(const Int *a, const Int *b, const Int *n) {
    Int r;
    Mult(a, b);
    Div(n, &r);
    Set(&r);
}

void Int::Mod(const Int *n) {
    Int r;
    Div(n, &r);
    Set(&r);
}

void Int::Div(const Int *a, Int *mod) {
    if(a->IsGreater(this)) {
        if(mod) mod->Set(this);
        CLEAR();
        return;
    }
    if(a->IsZero()) {
        printf("Divide by 0!\n");
        return;
    }
    if(IsEqual(a)) {
        if(mod) mod->CLEAR();
        Set(&_ONE);
        return;
    }

    Int rem(this);
    Int d(a);
    Int dq;
    CLEAR();

    uint32_t dSize = d.GetSize64();
    uint32_t tSize = rem.GetSize64();
    uint32_t qSize = tSize - dSize + 1;

    uint32_t shift = (uint32_t)LZC(d.bits64[dSize-1]);
    d.ShiftL(shift);
    rem.ShiftL(shift);

    uint64_t _dh = d.bits64[dSize-1];
    uint64_t _dl = (dSize > 1) ? d.bits64[dSize-2] : 0;
    int sb = tSize - 1;

    for(int j = 0; j < (int)qSize; j++) {
        uint64_t qhat = 0;
        uint64_t qrem = 0;
        bool skipCorrection = false;

        uint64_t nh = rem.bits64[sb - j + 1];
        uint64_t nm = rem.bits64[sb - j];

        if (nh == _dh) {
            qhat = ~0ULL;
            qrem = nh + nm;
            skipCorrection = (qrem < nh);
        } else {
            qhat = _udiv128(nh, nm, _dh, &qrem);
        }
        if(qhat == 0) continue;

        if(!skipCorrection) {
            uint64_t nl = rem.bits64[sb - j - 1];
            uint64_t estProH, estProL;
            estProL = _umul128(_dl, qhat, &estProH);
            if(isStrictGreater128(estProH, estProL, qrem, nl)) {
                qhat--;
                qrem += _dh;
                if(qrem >= _dh) {
                    estProL = _umul128(_dl, qhat, &estProH);
                    if(isStrictGreater128(estProH, estProL, qrem, nl)) {
                        qhat--;
                    }
                }
            }
        }

        dq.Mult(&d, qhat);
        rem.ShiftL64BitAndSub(&dq, qSize - j - 1);

        if(rem.IsNegative()) {
            rem.Add(&d);
            qhat--;
        }

        bits64[qSize - j - 1] = qhat;
    }

    if(mod) {
        rem.ShiftR(shift);
        mod->Set(&rem);
    }
}

void Int::GCD(const Int *a) {
    uint32_t k;
    uint32_t b;

    Int U(this);
    Int V(a);
    Int T;

    if(U.IsZero()) {
        Set(&V);
        return;
    }

    if(V.IsZero()) {
        Set(&U);
        return;
    }

    if(U.IsNegative()) U.Neg();
    if(V.IsNegative()) V.Neg();

    k = 0;
    while (U.GetBit(k) == 0 && V.GetBit(k) == 0)
        k++;
    U.ShiftR(k);
    V.ShiftR(k);
    if (U.GetBit(0) == 1) {
        T.Set(&V);
        T.Neg();
    } else {
        T.Set(&U);
    }

    do {
        if(T.IsNegative()) {
            T.Neg();
            b = 0; while(T.GetBit(b) == 0) b++;
            T.ShiftR(b);
            V.Set(&T);
            T.Set(&U);
        } else {
            b = 0; while(T.GetBit(b) == 0) b++;
            T.ShiftR(b);
            U.Set(&T);
        }
        T.Sub(&V);
    } while (!T.IsZero());

    Set(&U);
    ShiftL(k);
}

// -------------------
// Random operations
void Int::Rand(int nbit) {
    CLEAR();
    int nb64 = nbit / 64;
    int nb = nbit % 64;
    
    for(int i = 0; i < nb64; i++) {
        bits64[i] = ((uint64_t)rand() << 32) | rand();
    }
    
    if(nb > 0) {
        bits64[nb64] = ((uint64_t)rand() << 32) | rand();
        bits64[nb64] &= (1ULL << nb) - 1;
    }
}

void Int::Rand(const Int *randMax) {
    do {
        Rand(randMax->GetBitLength());
    } while(IsGreaterOrEqual(randMax));
}
