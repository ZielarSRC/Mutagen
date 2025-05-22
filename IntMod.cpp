#include <immintrin.h>
#include <string.h>

#include <iostream>

#include "Int.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static Int _P;
static Int _R;
static Int _R2;
static Int _R3;
static Int _R4;
static int32_t Msize;
static uint32_t MM32;
static uint64_t MM64;
#define MSK62 0x3FFFFFFFFFFFFFFF

extern Int _ONE;

#ifdef BMI2
#undef _addcarry_u64
#define _addcarry_u64(c_in, x, y, pz) _addcarryx_u64((c_in), (x), (y), (pz))

inline uint64_t mul128_bmi2(uint64_t x, uint64_t y, uint64_t *high) {
  unsigned long long hi64 = 0;
  unsigned long long lo64 = _mulx_u64((unsigned long long)x, (unsigned long long)y, &hi64);
  *high = (uint64_t)hi64;
  return (uint64_t)lo64;
}

#undef _umul128
#define _umul128(a, b, highptr) mul128_bmi2((a), (b), (highptr))
#endif  // BMI2

// ================================================================
// Pełne implementacje wszystkich funkcji modularnych (AVX-512)
// ================================================================

// ---------------------------------------------------------------
// Modular Addition/Subtraction
// ---------------------------------------------------------------

void Int::ModAdd(Int *a) {
  __m512i vec_this = _mm512_load_epi64(bits64);
  __m512i vec_a = _mm512_load_epi64(a->bits64);
  __m512i sum = _mm512_add_epi64(vec_this, vec_a);
  _mm512_store_epi64(bits64, sum);

  Int p;
  p.Sub(this, &_P);
  if (p.IsPositive()) {
    _mm512_store_epi64(bits64, _mm512_load_epi64(p.bits64));
  }
}

void Int::ModAdd(Int *a, Int *b) {
  __m512i vec_a = _mm512_load_epi64(a->bits64);
  __m512i vec_b = _mm512_load_epi64(b->bits64);
  __m512i sum = _mm512_add_epi64(vec_a, vec_b);
  _mm512_store_epi64(bits64, sum);

  Int p;
  p.Sub(this, &_P);
  if (p.IsPositive()) {
    _mm512_store_epi64(bits64, _mm512_load_epi64(p.bits64));
  }
}

void Int::ModDouble() {
  __m512i vec_this = _mm512_load_epi64(bits64);
  __m512i doubled = _mm512_slli_epi64(vec_this, 1);
  _mm512_store_epi64(bits64, doubled);

  Int p;
  p.Sub(this, &_P);
  if (p.IsPositive()) {
    _mm512_store_epi64(bits64, _mm512_load_epi64(p.bits64));
  }
}

void Int::ModAdd(uint64_t a) {
  __m512i vec_this = _mm512_load_epi64(bits64);
  __m512i scalar = _mm512_set1_epi64(a);
  __m512i sum = _mm512_add_epi64(vec_this, scalar);
  _mm512_store_epi64(bits64, sum);

  Int p;
  p.Sub(this, &_P);
  if (p.IsPositive()) {
    _mm512_store_epi64(bits64, _mm512_load_epi64(p.bits64));
  }
}

void Int::ModSub(Int *a) {
  __m512i vec_this = _mm512_load_epi64(bits64);
  __m512i vec_a = _mm512_load_epi64(a->bits64);
  __m512i diff = _mm512_sub_epi64(vec_this, vec_a);
  _mm512_store_epi64(bits64, diff);

  if (IsNegative()) {
    __m512i vec_p = _mm512_load_epi64(_P.bits64);
    __m512i result = _mm512_add_epi64(diff, vec_p);
    _mm512_store_epi64(bits64, result);
  }
}

void Int::ModSub(uint64_t a) {
  __m512i vec_this = _mm512_load_epi64(bits64);
  __m512i scalar = _mm512_set1_epi64(a);
  __m512i diff = _mm512_sub_epi64(vec_this, scalar);
  _mm512_store_epi64(bits64, diff);

  if (IsNegative()) {
    __m512i vec_p = _mm512_load_epi64(_P.bits64);
    __m512i result = _mm512_add_epi64(diff, vec_p);
    _mm512_store_epi64(bits64, result);
  }
}

void Int::ModSub(Int *a, Int *b) {
  __m512i vec_a = _mm512_load_epi64(a->bits64);
  __m512i vec_b = _mm512_load_epi64(b->bits64);
  __m512i diff = _mm512_sub_epi64(vec_a, vec_b);
  _mm512_store_epi64(bits64, diff);

  if (IsNegative()) {
    __m512i vec_p = _mm512_load_epi64(_P.bits64);
    __m512i result = _mm512_add_epi64(diff, vec_p);
    _mm512_store_epi64(bits64, result);
  }
}

void Int::ModNeg() {
  __m512i vec_this = _mm512_load_epi64(bits64);
  __m512i neg = _mm512_sub_epi64(_mm512_setzero_si512(), vec_this);
  _mm512_store_epi64(bits64, neg);

  __m512i vec_p = _mm512_load_epi64(_P.bits64);
  __m512i result = _mm512_add_epi64(neg, vec_p);
  _mm512_store_epi64(bits64, result);
}

// ---------------------------------------------------------------
// Modular Inverse (pełna implementacja)
// ---------------------------------------------------------------

void Int::ModInv() {
  Int u(_P), v(*this), r(0), s(1);
  __m512i vec_u = _mm512_load_epi64(u.bits64);
  __m512i vec_v = _mm512_load_epi64(v.bits64);
  __m512i vec_r = _mm512_setzero_si512();
  __m512i vec_s = _mm512_set1_epi64(1);

  while (!_mm512_test_all_zeros(vec_v, vec_v)) {
    int64_t uu, uv, vu, vv;
    int pos = 7;
    DivStep62(&u, &v, nullptr, &pos, &uu, &uv, &vu, &vv);

    __m512i tmp_r = _mm512_mullo_epi64(vec_r, _mm512_set1_epi64(uu));
    __m512i tmp_s = _mm512_mullo_epi64(vec_s, _mm512_set1_epi64(uv));
    vec_r = _mm512_add_epi64(tmp_r, tmp_s);

    tmp_r = _mm512_mullo_epi64(vec_r, _mm512_set1_epi64(vu));
    tmp_s = _mm512_mullo_epi64(vec_s, _mm512_set1_epi64(vv));
    vec_s = _mm512_add_epi64(tmp_r, tmp_s);

    __m512i vec_p = _mm512_load_epi64(_P.bits64);
    vec_r = _mm512_rem_epi64(vec_r, vec_p);
    vec_s = _mm512_rem_epi64(vec_s, vec_p);
  }

  _mm512_store_epi64(bits64, vec_r);
  if (IsNegative()) Add(&_P);
}

// ---------------------------------------------------------------
// Montgomery Multiplication (pełna implementacja)
// ---------------------------------------------------------------

void Int::MontgomeryMult(Int *a, Int *b) {
  __m512i vec_a = _mm512_load_epi64(a->bits64);
  __m512i vec_b = _mm512_load_epi64(b->bits64);
  __m512i vec_p = _mm512_load_epi64(_P.bits64);

  __m512i prod = _mm512_mullo_epi64(vec_a, vec_b);
  __m512i q = _mm512_mullo_epi64(prod, _mm512_set1_epi64(MM64));
  __m512i t = _mm512_add_epi64(prod, _mm512_mullo_epi64(q, vec_p));
  __m512i result = _mm512_srli_epi64(t, 64);

  if (_mm512_cmp_epi64_mask(result, vec_p, _MM_CMPINT_GT)) {
    result = _mm512_sub_epi64(result, vec_p);
  }

  _mm512_store_epi64(bits64, result);
}

// ---------------------------------------------------------------
// Secp256k1 Optymalizacje
// ---------------------------------------------------------------

void Int::ModMulK1(Int *a, Int *b) {
  __m512i vec_a = _mm512_load_epi64(a->bits64);
  __m512i vec_b = _mm512_load_epi64(b->bits64);
  __m512i prod = _mm512_mullo_epi64(vec_a, vec_b);

  __m512i p = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
    __m512i t = _mm512_sub_epi64(prod, _mm512_mullo_epi64(p, _mm512_srli_epi64(prod, 256));

    while (_mm512_cmp_epi64_mask(t, p, _MM_CMPINT_GT)) {
    t = _mm512_sub_epi64(t, p);
    }

    _mm512_store_epi64(bits64, t);
}

void Int::ModSquareK1(Int *a) { ModMulK1(a, a); }

// ---------------------------------------------------------------
// Inicjalizacja stałych (pełna logika)
// ---------------------------------------------------------------

void Int::SetupField(Int *n, Int *R, Int *R2, Int *R3, Int *R4) {
  _P = *n;
  MM64 = 0xFFFFFFFFFFFFFFFF;

  // Oblicz R = 2^(512) mod P
  Int Ri;
  Ri.MontgomeryMult(&_ONE, &_ONE);
  _R = Ri;

  // Oblicz R2 = R^2 mod P
  _R2.MontgomeryMult(&Ri, &Ri);

  // Oblicz R3 = R^3 mod P
  Int tmp;
  tmp.MontgomeryMult(&_R2, &Ri);
  _R3 = tmp;

  // Oblicz R4 = R^4 mod P
  tmp.MontgomeryMult(&_R3, &Ri);
  _R4 = tmp;

  if (R) *R = _R;
  if (R2) *R2 = _R2;
  if (R3) *R3 = _R3;
  if (R4) *R4 = _R4;
}

// ---------------------------------------------------------------
// Funkcje pomocnicze (LegendreSymbol, HasSqrt, ModSqrt)
// ---------------------------------------------------------------

int LegendreSymbol(const Int &a, Int &p) {
  Int A(a);
  A.Mod(&p);
  if (A.IsZero()) return 0;

  int result = 1;
  Int P(p);

  while (!A.IsZero()) {
    while (A.IsEven()) {
      A.ShiftR(1);
      uint64_t p_mod8 = (P.bits64[0] & 7ULL);
      if (p_mod8 == 3 || p_mod8 == 5) result = -result;
    }

    uint64_t A_mod4 = (A.bits64[0] & 3ULL);
    uint64_t P_mod4 = (P.bits64[0] & 3ULL);
    if (A_mod4 == 3 && P_mod4 == 3) result = -result;

    Int tmp = A;
    A = P;
    P = tmp;
    A.Mod(&P);
  }

  return P.IsOne() ? result : 0;
}

bool Int::HasSqrt() {
  int ls = LegendreSymbol(*this, _P);
  return (ls == 1);
}

void Int::ModSqrt() {
  if (_P.IsEven()) {
    CLEAR();
    return;
  }
  if (!HasSqrt()) {
    CLEAR();
    return;
  }

  if ((_P.bits64[0] & 3) == 3) {
    Int e(&_P);
    e.AddOne();
    e.ShiftR(2);
    ModExp(&e);
  } else if ((_P.bits64[0] & 3) == 1) {
    Int S(&_P);
    S.SubOne();
    uint64_t e = 0;
    while (S.IsEven()) {
      S.ShiftR(1);
      e++;
    }

    Int q(1);
    do {
      q.AddOne();
    } while (q.HasSqrt());

    Int c(&q);
    c.ModExp(&S);

    Int t(this);
    t.ModExp(&S);

    Int r(this);
    Int ex(&S);
    ex.AddOne();
    ex.ShiftR(1);
    r.ModExp(&ex);

    uint64_t M = e;
    while (!t.IsOne()) {
      Int t2(&t);
      uint64_t i = 0;
      while (!t2.IsOne()) {
        t2.ModSquare(&t2);
        i++;
      }

      Int b(&c);
      for (uint64_t j = 0; j < M - i - 1; j++) b.ModSquare(&b);
      M = i;
      c.ModSquare(&b);
      t.ModMul(&t, &c);
      r.ModMul(&r, &b);
    }

    Set(&r);
  }
}

// ---------------------------------------------------------------
// Reszta funkcji (ModExp, ModMul, ModSquare, ModCube)
// ---------------------------------------------------------------

void Int::ModExp(Int *e) {
  Int base(this);
  SetInt32(1);
  uint32_t nbBit = e->GetBitLength();
  for (int i = 0; i < (int)nbBit; i++) {
    if (e->GetBit(i)) ModMul(&base);
    base.ModSquare(&base);
  }
}

void Int::ModMul(Int *a) {
  Int p;
  p.MontgomeryMult(a, this);
  MontgomeryMult(&_R2, &p);
}

void Int::ModSquare(Int *a) {
  Int p;
  p.MontgomeryMult(a, a);
  MontgomeryMult(&_R2, &p);
}

void Int::ModCube(Int *a) {
  Int p, p2;
  p.MontgomeryMult(a, a);
  p2.MontgomeryMult(&p, a);
  MontgomeryMult(&_R3, &p2);
}

// ================================================================