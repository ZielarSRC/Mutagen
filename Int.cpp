#include <immintrin.h>
#include <math.h>
#include <string.h>

#include "Int.h"
#include "IntGroup.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

Int _ONE((uint64_t)1);
Int Int::P;

// AVX-512 Optimized Addition ==========================================
void Int::AddAVX512(Int *a) {
  __m512i carry = _mm512_setzero_si512();
  for (int i = 0; i < NB64BLOCK; i += 8) {
    __m512i val = _mm512_load_epi64(&bits64[i]);
    __m512i add = _mm512_load_epi64(&a->bits64[i]);
    __m512i res = _mm512_add_epi64(val, add);
    res = _mm512_add_epi64(res, carry);
    carry = _mm512_srli_epi64(_mm512_cmpgt_epi64_mask(res, val), 63);
    _mm512_store_epi64(&bits64[i], res);
  }
}

// AVX-512 Montgomery Multiplication ===================================
void Int::MontgomeryMultAVX512(Int *a, Int *b, Int *n) {
  __m512i R = _mm512_set1_epi64(1ULL << 63);  // R = 2^64
  __m512i NI =
      _mm512_set1_epi64(n->MontgomeryInverse());  // Precomputed n^-1 mod R

  alignas(64) __m512i T[NB64BLOCK / 8];
  __m512i carry = _mm512_setzero_si512();

  for (int i = 0; i < NB64BLOCK; i++) {
    __m512i ai = _mm512_set1_epi64(a->bits64[i]);
    __m512i bi = _mm512_load_epi64(&b->bits64[0]);
    __m512i ti = _mm512_add_epi64(_mm512_mullo_epi64(ai, bi), carry);
    carry = _mm512_srli_epi64(ti, 64);
    T[i] = _mm512_and_si512(ti, _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
  }

  // Reduction phase using AVX-512
  for (int i = 0; i < NB64BLOCK; i++) {
    __m512i m = _mm512_mullo_epi64(T[i], NI);
    __m512i q = _mm512_add_epi64(
        _mm512_mullo_epi64(m, _mm512_load_epi64(n->bits64)), T[i]);
    carry = _mm512_srli_epi64(q, 64);
    if (i < NB64BLOCK - 1) T[i + 1] = _mm512_add_epi64(T[i + 1], carry);
  }

  // Final subtraction of modulus
  __mmask8 cmp =
      _mm512_cmpgt_epi64_mask(T[NB64BLOCK - 1], _mm512_load_epi64(n->bits64));
  if (cmp)
    _mm512_store_epi64(bits64, _mm512_sub_epi64(T[NB64BLOCK - 1],
                                                _mm512_load_epi64(n->bits64)));
  else
    _mm512_store_epi64(bits64, T[NB64BLOCK - 1]);
}

// Original implementation below (unchanged) ===========================
Int::Int() { CLEAR(); }

Int::Int(Int *a) {
  if (a)
    Set(a);
  else
    CLEAR();
}

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

// ------------------------------------------------

void Int::CLEAR() { memset(bits64, 0, NB64BLOCK * 8); }

void Int::CLEARFF() { memset(bits64, 0xFF, NB64BLOCK * 8); }

// ------------------------------------------------

void Int::Set(Int *a) {
  for (int i = 0; i < NB64BLOCK; i++) bits64[i] = a->bits64[i];
}

// ------------------------------------------------

void Int::Add(Int *a) {
  uint64_t acc0 = bits64[0];
  uint64_t acc1 = bits64[1];
  uint64_t acc2 = bits64[2];
  uint64_t acc3 = bits64[3];
  uint64_t acc4 = bits64[4];

#if NB64BLOCK > 5
  uint64_t acc5 = bits64[5];
  uint64_t acc6 = bits64[6];
  uint64_t acc7 = bits64[7];
  uint64_t acc8 = bits64[8];
#endif

  asm("add %[src0], %[dst0]    \n\t"
      "adc %[src1], %[dst1]    \n\t"
      "adc %[src2], %[dst2]    \n\t"
      "adc %[src3], %[dst3]    \n\t"
      "adc %[src4], %[dst4]    \n\t"
#if NB64BLOCK > 5

      "adc %[src5], %[dst5]    \n\t"
      "adc %[src6], %[dst6]    \n\t"
      "adc %[src7], %[dst7]    \n\t"
      "adc %[src8], %[dst8]    \n\t"
#endif

      : [dst0] "+r"(acc0), [dst1] "+r"(acc1), [dst2] "+r"(acc2),
        [dst3] "+r"(acc3), [dst4] "+r"(acc4)
#if NB64BLOCK > 5
                               ,
        [dst5] "+r"(acc5), [dst6] "+r"(acc6), [dst7] "+r"(acc7),
        [dst8] "+r"(acc8)
#endif

      : [src0] "r"(a->bits64[0]), [src1] "r"(a->bits64[1]),
        [src2] "r"(a->bits64[2]), [src3] "r"(a->bits64[3]),
        [src4] "r"(a->bits64[4])
#if NB64BLOCK > 5
            ,
        [src5] "r"(a->bits64[5]), [src6] "r"(a->bits64[6]),
        [src7] "r"(a->bits64[7]), [src8] "r"(a->bits64[8])
#endif
      : "cc");

  bits64[0] = acc0;
  bits64[1] = acc1;
  bits64[2] = acc2;
  bits64[3] = acc3;
  bits64[4] = acc4;

#if NB64BLOCK > 5
  bits64[5] = acc5;
  bits64[6] = acc6;
  bits64[7] = acc7;
  bits64[8] = acc8;
#endif
}

// ------------------------------------------------

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

// ------------------------------------------------
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

// ------------------------------------------------

void Int::Add(Int *a, Int *b) {
  uint64_t acc0 = a->bits64[0];
  uint64_t acc1 = a->bits64[1];
  uint64_t acc2 = a->bits64[2];
  uint64_t acc3 = a->bits64[3];
  uint64_t acc4 = a->bits64[4];

#if NB64BLOCK > 5
  uint64_t acc5 = a->bits64[5];
  uint64_t acc6 = a->bits64[6];
  uint64_t acc7 = a->bits64[7];
  uint64_t acc8 = a->bits64[8];
#endif

  asm("add %[b0], %[a0]       \n\t"
      "adc %[b1], %[a1]       \n\t"
      "adc %[b2], %[a2]       \n\t"
      "adc %[b3], %[a3]       \n\t"
      "adc %[b4], %[a4]       \n\t"
#if NB64BLOCK > 5
      "adc %[b5], %[a5]       \n\t"
      "adc %[b6], %[a6]       \n\t"
      "adc %[b7], %[a7]       \n\t"
      "adc %[b8], %[a8]       \n\t"
#endif
      : [a0] "+r"(acc0), [a1] "+r"(acc1), [a2] "+r"(acc2), [a3] "+r"(acc3),
        [a4] "+r"(acc4)
#if NB64BLOCK > 5
            ,
        [a5] "+r"(acc5), [a6] "+r"(acc6), [a7] "+r"(acc7), [a8] "+r"(acc8)
#endif
      : [b0] "r"(b->bits64[0]), [b1] "r"(b->bits64[1]), [b2] "r"(b->bits64[2]),
        [b3] "r"(b->bits64[3]), [b4] "r"(b->bits64[4])
#if NB64BLOCK > 5
                                    ,
        [b5] "r"(b->bits64[5]), [b6] "r"(b->bits64[6]), [b7] "r"(b->bits64[7]),
        [b8] "r"(b->bits64[8])
#endif
      : "cc");

  bits64[0] = acc0;
  bits64[1] = acc1;
  bits64[2] = acc2;
  bits64[3] = acc3;
  bits64[4] = acc4;

#if NB64BLOCK > 5
  bits64[5] = acc5;
  bits64[6] = acc6;
  bits64[7] = acc7;
  bits64[8] = acc8;
#endif
}

// ------------------------------------------------

uint64_t Int::AddCh(Int *a, uint64_t ca, Int *b, uint64_t cb) {
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

uint64_t Int::AddCh(Int *a, uint64_t ca) {
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

// ------------------------------------------------

uint64_t Int::AddC(Int *a) {
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

// ------------------------------------------------

void Int::AddAndShift(Int *a, Int *b, uint64_t cH) {
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

// ------------------------------------------------

void Int::MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21,
                       int64_t _22, uint64_t *cu, uint64_t *cv) {
  Int t1, t2, t3, t4;
  uint64_t c1, c2, c3, c4;
  c1 = t1.IMult(u, _11);
  c2 = t2.IMult(v, _12);
  c3 = t3.IMult(u, _21);
  c4 = t4.IMult(v, _22);
  *cu = u->AddCh(&t1, c1, &t2, c2);
  *cv = v->AddCh(&t3, c3, &t4, c4);
}

void Int::MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21,
                       int64_t _22) {
  Int t1, t2, t3, t4;
  t1.IMult(u, _11);
  t2.IMult(v, _12);
  t3.IMult(u, _21);
  t4.IMult(v, _22);
  u->Add(&t1, &t2);
  v->Add(&t3, &t4);
}

// ------------------------------------------------

bool Int::IsGreater(Int *a) {
  int i;

  for (i = NB64BLOCK - 1; i >= 0;) {
    if (a->bits64[i] != bits64[i]) break;
    i--;
  }

  if (i >= 0) {
    return bits64[i] > a->bits64[i];
  } else {
    return false;
  }
}

// ------------------------------------------------

bool Int::IsLower(Int *a) {
  int i;

  for (i = NB64BLOCK - 1; i >= 0;) {
    if (a->bits64[i] != bits64[i]) break;
    i--;
  }

  if (i >= 0) {
    return bits64[i] < a->bits64[i];
  } else {
    return false;
  }
}

// ------------------------------------------------

bool Int::IsGreaterOrEqual(Int *a) {
  Int p;
  p.Sub(this, a);
  return p.IsPositive();
}

// ------------------------------------------------

bool Int::IsLowerOrEqual(Int *a) {
  int i = NB64BLOCK - 1;

  while (i >= 0) {
    if (a->bits64[i] != bits64[i]) break;
    i--;
  }

  if (i >= 0) {
    return bits64[i] < a->bits64[i];
  } else {
    return true;
  }
}

bool Int::IsEqual(Int *a) {
  return

#if NB64BLOCK > 5
      (bits64[8] == a->bits64[8]) && (bits64[7] == a->bits64[7]) &&
      (bits64[6] == a->bits64[6]) && (bits64[5] == a->bits64[5]) &&
#endif
      (bits64[4] == a->bits64[4]) && (bits64[3] == a->bits64[3]) &&
      (bits64[2] == a->bits64[2]) && (bits64[1] == a->bits64[1]) &&
      (bits64[0] == a->bits64[0]);
}

bool Int::IsOne() { return IsEqual(&_ONE); }

bool Int::IsZero() {
#if NB64BLOCK > 5
  return (bits64[8] | bits64[7] | bits64[6] | bits64[5] | bits64[4] |
          bits64[3] | bits64[2] | bits64[1] | bits64[0]) == 0;
#else
  return (bits64[4] | bits64[3] | bits64[2] | bits64[1] | bits64[0]) == 0;
#endif
}

// ------------------------------------------------

void Int::SetInt32(uint32_t value) {
  CLEAR();
  bits[0] = value;
}

// ------------------------------------------------

uint32_t Int::GetInt32() { return bits[0]; }

// ------------------------------------------------

unsigned char Int::GetByte(int n) {
  unsigned char *bbPtr = (unsigned char *)bits;
  return bbPtr[n];
}

void Int::Set32Bytes(unsigned char *bytes) {
  CLEAR();
  uint64_t *ptr = (uint64_t *)bytes;
  bits64[3] = _byteswap_uint64(ptr[0]);
  bits64[2] = _byteswap_uint64(ptr[1]);
  bits64[1] = _byteswap_uint64(ptr[2]);
  bits64[0] = _byteswap_uint64(ptr[3]);
}

void Int::Get32Bytes(unsigned char *buff) {
  uint64_t *ptr = (uint64_t *)buff;
  ptr[3] = _byteswap_uint64(bits64[0]);
  ptr[2] = _byteswap_uint64(bits64[1]);
  ptr[1] = _byteswap_uint64(bits64[2]);
  ptr[0] = _byteswap_uint64(bits64[3]);
}

// ------------------------------------------------

void Int::SetByte(int n, unsigned char byte) {
  unsigned char *bbPtr = (unsigned char *)bits;
  bbPtr[n] = byte;
}

// ------------------------------------------------

void Int::SetDWord(int n, uint32_t b) { bits[n] = b; }

// ------------------------------------------------

void Int::SetQWord(int n, uint64_t b) { bits64[n] = b; }

// ------------------------------------------------

void Int::Sub(Int *a) {
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

// ------------------------------------------------

void Int::Sub(Int *a, Int *b) {
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

// ------------------------------------------------

void Int::Neg() {
  for (int i = 0; i < NB64BLOCK; i++) bits64[i] = ~bits64[i];
  AddOne();
}

// ------------------------------------------------

void Int::Abs() {
  if (IsNegative()) Neg();
}

// ------------------------------------------------

bool Int::IsPositive() {
  return ((int64_t)bits64[NB64BLOCK - 1]) >= 0;
}

bool Int::IsNegative() {
  return ((int64_t)bits64[NB64BLOCK - 1]) < 0;
}

bool Int::IsStrictPositive() {
  return IsPositive() && !IsZero();
}

bool Int::IsEven() {
  return (bits64[0] & 1) == 0;
}

bool Int::IsOdd() {
  return (bits64[0] & 1) == 1;
}

// ------------------------------------------------

int Int::GetBit(uint32_t n) {
  uint32_t w = n / 64;
  uint32_t b = n % 64;
  return (bits64[w] >> b) & 1;
}

// ------------------------------------------------

void Int::ShiftL(uint32_t n) {
  if (n == 0) return;
  uint32_t d = n / 64;
  uint32_t s = n % 64;

  if (d > 0) {
    for (int i = NB64BLOCK - 1; i >= (int)d; i--) bits64[i] = bits64[i - d];
    for (uint32_t i = 0; i < d; i++) bits64[i] = 0;
  }

  if (s != 0) {
    for (int i = NB64BLOCK - 1; i > 0; i--)
      bits64[i] = (bits64[i] << s) | (bits64[i - 1] >> (64 - s));
    bits64[0] <<= s;
  }
}

// ------------------------------------------------

void Int::ShiftR(uint32_t n) {
  if (n == 0) return;
  uint32_t d = n / 64;
  uint32_t s = n % 64;

  if (d > 0) {
    for (int i = 0; i < NB64BLOCK - (int)d; i++) bits64[i] = bits64[i + d];
    for (int i = NB64BLOCK - d; i < NB64BLOCK; i++) bits64[i] = 0;
  }

  if (s != 0) {
    for (int i = 0; i < NB64BLOCK - 1; i++)
      bits64[i] = (bits64[i] >> s) | (bits64[i + 1] << (64 - s));
    bits64[NB64BLOCK - 1] >>= s;
  }
}

// ------------------------------------------------

void Int::SwapBit(int bitNumber) {
  uint32_t idx = bitNumber / 64;
  uint32_t pos = bitNumber % 64;
  bits64[idx] ^= (1ULL << pos);
}

// ------------------------------------------------

void Int::MaskByte(int n) {
  unsigned char *bbPtr = (unsigned char *)bits;
  for (int i = n; i < NB32BLOCK * 4; ++i) bbPtr[i] = 0;
}

// ------------------------------------------------

int Int::GetBitLength() {
  for (int i = NB64BLOCK - 1; i >= 0; i--) {
    if (bits64[i]) {
      int lz = __builtin_clzll(bits64[i]);
      return (i + 1) * 64 - lz;
    }
  }
  return 0;
}

// ------------------------------------------------

int Int::GetLowestBit() {
  for (int i = 0; i < NB64BLOCK; i++) {
    if (bits64[i])
      return i * 64 + __builtin_ctzll(bits64[i]);
  }
  return -1;
}

// ------------------------------------------------

int Int::GetSize() {
  int i = NB32BLOCK - 1;
  while (i > 0 && bits[i] == 0) i--;
  return i + 1;
}

int Int::GetSize64() {
  int i = NB64BLOCK - 1;
  while (i > 0 && bits64[i] == 0) i--;
  return i + 1;
}

// ------------------------------------------------

void Int::Rand(int nbit) {
  CLEAR();
  int nbytes = (nbit + 7) / 8;
  unsigned char *bbPtr = (unsigned char *)bits;
  for (int i = 0; i < nbytes; i++) bbPtr[i] = rand() & 0xFF;
  if (nbit % 8) bbPtr[nbytes - 1] &= (1 << (nbit % 8)) - 1;
}

void Int::Rand(Int *randMax) {
  int nbit = randMax->GetBitLength();
  do {
    Rand(nbit);
  } while (IsGreaterOrEqual(randMax));
}

// ------------------------------------------------

void Int::SetBase10(char *value) {
  CLEAR();
  int isNegative = 0;
  char *p = value;
  if (*p == '-') {
    isNegative = 1;
    p++;
  }
  while (*p) {
    Mult(10);
    Add((uint64_t)(*p - '0'));
    p++;
  }
  if (isNegative) Neg();
}

std::string Int::GetBase10() {
  std::string ret;
  Int N(this);
  int isNegative = N.IsNegative();
  if (isNegative) N.Neg();

  unsigned char digits[1024] = {0};
  int digitslen = 1;
  for (int i = 0; i < NB64BLOCK * 8; i++) {
    unsigned int carry = N.GetByte(NB64BLOCK * 8 - i - 1);
    for (int j = 0; j < digitslen; j++) {
      carry += (unsigned int)(digits[j]) << 8;
      digits[j] = (unsigned char)(carry % 10);
      carry /= 10;
    }
    while (carry > 0) {
      digits[digitslen++] = (unsigned char)(carry % 10);
      carry /= 10;
    }
  }

  if (isNegative) ret.push_back('-');
  for (int i = 0; i < digitslen; i++)
    ret.push_back((char)('0' + digits[digitslen - 1 - i]));
  if (ret.length() == 0) ret.push_back('0');
  return ret;
}

// ------------------------------------------------

std::string Int::GetBase16() {
  std::string ret;
  char charset[] = "0123456789ABCDEF";
  Int N(this);
  int isNegative = N.IsNegative();
  if (isNegative) N.Neg();

  unsigned char digits[1024] = {0};
  int digitslen = 1;
  for (int i = 0; i < NB64BLOCK * 8; i++) {
    unsigned int carry = N.GetByte(NB64BLOCK * 8 - i - 1);
    for (int j = 0; j < digitslen; j++) {
      carry += (unsigned int)(digits[j]) << 8;
      digits[j] = (unsigned char)(carry % 16);
      carry /= 16;
    }
    while (carry > 0) {
      digits[digitslen++] = (unsigned char)(carry % 16);
      carry /= 16;
    }
  }

  if (isNegative) ret.push_back('-');
  for (int i = 0; i < digitslen; i++)
    ret.push_back(charset[digits[digitslen - 1 - i]]);

  if (ret.length() == 0) ret.push_back('0');
  return ret;
}

// ------------------------------------------------

std::string Int::GetBase2() {
  std::string ret;
  Int N(this);
  int isNegative = N.IsNegative();
  if (isNegative) N.Neg();

  unsigned char digits[1024] = {0};
  int digitslen = 1;
  for (int i = 0; i < NB64BLOCK * 8; i++) {
    unsigned int carry = N.GetByte(NB64BLOCK * 8 - i - 1);
    for (int j = 0; j < digitslen; j++) {
      carry += (unsigned int)(digits[j]) << 8;
      digits[j] = (unsigned char)(carry % 2);
      carry /= 2;
    }
    while (carry > 0) {
      digits[digitslen++] = (unsigned char)(carry % 2);
      carry /= 2;
    }
  }

  if (isNegative) ret.push_back('-');
  for (int i = 0; i < digitslen; i++)
    ret.push_back((char)('0' + digits[digitslen - 1 - i]));
  if (ret.length() == 0) ret.push_back('0');
  return ret;
}

std::string Int::GetBaseN(int n, char *charset) {
  std::string ret;

  Int N(this);
  int isNegative = N.IsNegative();
  if (isNegative) N.Neg();

  unsigned char digits[1024] = {0};
  int digitslen = 1;
  for (int i = 0; i < NB64BLOCK * 8; i++) {
    unsigned int carry = N.GetByte(NB64BLOCK * 8 - i - 1);
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

  if (isNegative) ret.push_back('-');
  for (int i = 0; i < digitslen; i++)
    ret.push_back(charset[digits[digitslen - 1 - i]]);
  if (ret.length() == 0) ret.push_back('0');
  return ret;
}

// ------------------------------------------------

double Int::ToDouble() {
  double val = 0;
  double factor = 1.0;
  for (int i = 0; i < NB64BLOCK; i++) {
    val += (double)bits64[i] * factor;
    factor *= 18446744073709551616.0; // 2^64
  }
  return val;
}

// ------------------------------------------------

uint64_t Int::IMult(Int *a, int64_t b) {
  uint64_t carry = 0;
  for (int i = 0; i < NB64BLOCK; i++)
    carry = _mulx_u64(a->bits64[i], b, bits64 + i);
  return carry;
}

void Int::Mult(Int *a, Int *b) {
  CLEAR();
  Int tmp;
  for (int i = 0; i < NB64BLOCK; i++) {
    if (b->bits64[i]) {
      tmp.IMult(a, b->bits64[i]);
      ShiftL(i * 64);
      Add(&tmp);
      ShiftR(i * 64);
    }
  }
}

void Int::Mult(uint64_t b) {
  uint64_t carry = 0;
  for (int i = 0; i < NB64BLOCK; i++) {
    unsigned __int128 res = (unsigned __int128)bits64[i] * b + carry;
    bits64[i] = (uint64_t)res;
    carry = (uint64_t)(res >> 64);
  }
}

void Int::Mult(Int *a, uint64_t b) {
  uint64_t carry = 0;
  for (int i = 0; i < NB64BLOCK; i++) {
    unsigned __int128 res = (unsigned __int128)a->bits64[i] * b + carry;
    bits64[i] = (uint64_t)res;
    carry = (uint64_t)(res >> 64);
  }
}

// ------------------------------------------------

void Int::Div(Int *a, Int *b, Int *rem) {
  Int Q, R, D, tmp;
  Q.CLEAR();
  R.CLEAR();
  D.Set(b);
  int n = a->GetBitLength();
  for (int i = n - 1; i >= 0; i--) {
    R.ShiftL(1);
    R.bits64[0] |= a->GetBit(i);
    if (!R.IsLower(&D)) {
      R.Sub(&D);
      Q.SetBit(i);
    }
  }
  if (rem) rem->Set(&R);
  Set(&Q);
}

// ------------------------------------------------

void Int::Mod(Int *a, Int *m) {
  Int tmp;
  Div(a, m, &tmp);
  Set(&tmp);
}

void Int::InvMod(Int *a, Int *m) {
  Int t, newt, r, newr, quotient, temp;
  t.CLEAR();
  newt = _ONE;
  r.Set(m);
  newr.Set(a);

  while (!newr.IsZero()) {
    quotient.Div(&r, &newr, nullptr);

    temp.Set(&newt);
    temp.Mult(&quotient, &newt);
    t.Sub(&temp);

    temp.Set(&newr);
    temp.Mult(&quotient, &newr);
    r.Sub(&temp);

    t.Set(&newt);
    r.Set(&newr);
  }

  if (r.bits64[0] > 1) CLEAR();
  if (t.IsNegative()) t.Add(m);
  Set(&t);
}

// ------------------------------------------------

void Int::GCD(Int *a, Int *b) {
  Int A(a), B(b);
  while (!B.IsZero()) {
    Int tmp;
    tmp.Set(&B);
    A.Mod(&A, &B);
    A.Set(&B);
    B.Set(&tmp);
  }
  Set(&A);
}

// ------------------------------------------------

void Int::MontgomeryPrepare(Int *mod) {
  // Calculate R = 2^(NB64BLOCK*64)
  CLEAR();
  bits64[NB64BLOCK - 1] = 1;
  Mod(this, mod);
}

uint64_t Int::MontgomeryInverse() {
  // Calculate -mod^{-1} mod 2^64 using extended Euclidean algorithm
  uint64_t t = 0, r = 0, newt = 1, newr = bits64[0];
  for (int i = 0; i < 64; i++) {
    if (!(newr & 1)) {
      newr >>= 1;
      newt <<= 1;
    } else {
      if (newt < newr) {
        t = newt;
        newt = newr;
        newr = t;
      }
      newt = newt - newr;
      newt >>= 1;
    }
  }
  return -newt;
}

// ------------------------------------------------

void Int::MontgomeryMult(Int *a, Int *b, Int *mod) {
  Int t;
  t.CLEAR();
  for (int i = 0; i < NB64BLOCK; i++) {
    uint64_t u = t.bits64[0] + a->bits64[i] * b->bits64[0];
    u *= mod->MontgomeryInverse();
    t.Add(a, b);
    t.Add(mod, u);
    t.ShiftR(64);
  }
  if (!t.IsLower(mod)) t.Sub(mod);
  Set(&t);
}

void Int::MontgomeryReduce(Int *t, Int *mod) {
  for (int i = 0; i < NB64BLOCK; i++) {
    uint64_t u = t->bits64[0] * mod->MontgomeryInverse();
    t->Add(mod, u);
    t->ShiftR(64);
  }
  if (!t->IsLower(mod)) t->Sub(mod);
  Set(t);
}

// ------------------------------------------------

void Int::PowMod(Int *a, Int *exp, Int *mod) {
  Int x, y;
  x.Set(a);
  y = _ONE;
  for (int i = exp->GetBitLength() - 1; i >= 0; i--) {
    y.MontgomeryMult(&y, &y, mod);
    if (exp->GetBit(i)) y.MontgomeryMult(&y, &x, mod);
  }
  Set(&y);
}

void Int::SetBit(int n) {
  int q = n / 64;
  int r = n % 64;
  bits64[q] |= (1ULL << r);
}

void Int::ClearBit(int n) {
  int q = n / 64;
  int r = n % 64;
  bits64[q] &= ~(1ULL << r);
}

// ------------------------------------------------

void Int::Mult(uint64_t b) {
  uint64_t carry = 0;
  for (int i = 0; i < NB64BLOCK; i++) {
    unsigned __int128 res = (unsigned __int128)bits64[i] * b + carry;
    bits64[i] = (uint64_t)res;
    carry = (uint64_t)(res >> 64);
  }
}

void Int::Mult(Int *a, uint64_t b) {
  uint64_t carry = 0;
  for (int i = 0; i < NB64BLOCK; i++) {
    unsigned __int128 res = (unsigned __int128)a->bits64[i] * b + carry;
    bits64[i] = (uint64_t)res;
    carry = (uint64_t)(res >> 64);
  }
}

// ------------------------------------------------

void Int::MontgomeryPrepare(Int *mod) {
  // Calculate R = 2^(NB64BLOCK*64)
  CLEAR();
  bits64[NB64BLOCK - 1] = 1;
  Mod(this, mod);
}

uint64_t Int::MontgomeryInverse() {
  // Calculate -mod^{-1} mod 2^64 using extended Euclidean algorithm
  uint64_t t = 0, r = 0, newt = 1, newr = bits64[0];
  for (int i = 0; i < 64; i++) {
    if (!(newr & 1)) {
      newr >>= 1;
      newt <<= 1;
    } else {
      if (newt < newr) {
        t = newt;
        newt = newr;
        newr = t;
      }
      newt = newt - newr;
      newt >>= 1;
    }
  }
  return -newt;
}

// ------------------------------------------------

void Int::MontgomeryMult(Int *a, Int *b, Int *mod) {
  Int t;
  t.CLEAR();
  for (int i = 0; i < NB64BLOCK; i++) {
    uint64_t u = t.bits64[0] + a->bits64[i] * b->bits64[0];
    u *= mod->MontgomeryInverse();
    t.Add(a, b);
    t.Add(mod, u);
    t.ShiftR(64);
  }
  if (!t.IsLower(mod)) t.Sub(mod);
  Set(&t);
}

void Int::MontgomeryReduce(Int *t, Int *mod) {
  for (int i = 0; i < NB64BLOCK; i++) {
    uint64_t u = t->bits64[0] * mod->MontgomeryInverse();
    t->Add(mod, u);
    t->ShiftR(64);
  }
  if (!t->IsLower(mod)) t->Sub(mod);
  Set(t);
}

// ------------------------------------------------

void Int::PowMod(Int *a, Int *exp, Int *mod) {
  Int x, y;
  x.Set(a);
  y = _ONE;
  for (int i = exp->GetBitLength() - 1; i >= 0; i--) {
    y.MontgomeryMult(&y, &y, mod);
    if (exp->GetBit(i)) y.MontgomeryMult(&y, &x, mod);
  }
  Set(&y);
}

void Int::GCD(Int *a, Int *b) {
  Int A(a), B(b);
  while (!B.IsZero()) {
    Int tmp;
    tmp.Set(&B);
    A.Mod(&A, &B);
    A.Set(&B);
    B.Set(&tmp);
  }
  Set(&A);
}

// EOF
