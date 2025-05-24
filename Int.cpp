#include "Int.h"
#include "IntGroup.h"
#include <string.h>
#include <math.h>
#include <emmintrin.h>
#include <cctype>
#include <cstdio>

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

Int _ONE((uint64_t)1);

Int Int::P;

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
  memset(bits64,0, NB64BLOCK*8);
}

void Int::CLEARFF() {
  memset(bits64, 0xFF, NB64BLOCK * 8);
}

// -------------------
void Int::Set(const Int *a) {
  for (int i = 0; i<NB64BLOCK; i++)
    bits64[i] = a->bits64[i];
}

// -------------------
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
  c = _addcarry_u64(c, bits64[0],1, bits64 +0);
  c = _addcarry_u64(c, bits64[1],0, bits64 +1);
  c = _addcarry_u64(c, bits64[2],0, bits64 +2);
  c = _addcarry_u64(c, bits64[3],0, bits64 +3);
  c = _addcarry_u64(c, bits64[4],0, bits64 +4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5],0, bits64 +5);
  c = _addcarry_u64(c, bits64[6],0, bits64 +6);
  c = _addcarry_u64(c, bits64[7],0, bits64 +7);
  c = _addcarry_u64(c, bits64[8],0, bits64 +8);
#endif
}

// -------------------
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
  c = _subborrow_u64(c, bits64[0], a->bits64[0], bits64 +0);
  c = _subborrow_u64(c, bits64[1], a->bits64[1], bits64 +1);
  c = _subborrow_u64(c, bits64[2], a->bits64[2], bits64 +2);
  c = _subborrow_u64(c, bits64[3], a->bits64[3], bits64 +3);
  c = _subborrow_u64(c, bits64[4], a->bits64[4], bits64 +4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], a->bits64[5], bits64 +5);
  c = _subborrow_u64(c, bits64[6], a->bits64[6], bits64 +6);
  c = _subborrow_u64(c, bits64[7], a->bits64[7], bits64 +7);
  c = _subborrow_u64(c, bits64[8], a->bits64[8], bits64 +8);
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

// -------------------
void Int::Xor(const Int *a) {
  if (!a) return;
  uint64_t* this_bits = bits64;
  const uint64_t* a_bits = a->bits64;
  const int count = NB64BLOCK;
#if defined(__AVX512F__)
  // Use AVX512 for Xeon Platinum
  int i = 0;
  for (; i + 4 <= count; i += 4) {
    __m256i x = _mm256_load_si256((const __m256i*)(a_bits + i));
    __m256i y = _mm256_load_si256((const __m256i*)(this_bits + i));
    __m256i z = _mm256_xor_si256(x, y);
    _mm256_store_si256((__m256i*)(this_bits + i), z);
  }
  for (; i < count; ++i) {
    this_bits[i] ^= a_bits[i];
  }
#else
  for (int i = 0; i < count; ++i) {
    this_bits[i] ^= a_bits[i];
  }
#endif
}

// -------------------
bool Int::IsGreater(const Int *a) const {
  int i = NB64BLOCK-1;
  for(;i>=0;--i) {
    if(a->bits64[i]!= bits64[i])
      break;
  }
  if(i>=0) {
    return bits64[i]>a->bits64[i];
  } else {
    return false;
  }
}

bool Int::IsLower(const Int *a) const {
  int i = NB64BLOCK-1;
  for(;i>=0;--i) {
    if(a->bits64[i]!= bits64[i])
      break;
  }
  if(i>=0) {
    return bits64[i]<a->bits64[i];
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
  for(;i>=0;--i) {
    if(a->bits64[i]!= bits64[i])
      break;
  }
  if(i>=0) {
    return bits64[i]<a->bits64[i];
  } else {
    return true;
  }
}

bool Int::IsEqual(const Int *a) const {
#if NB64BLOCK > 5
  return (bits64[8] == a->bits64[8]) &&
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
bool Int::IsPositive() const { return (int64_t)(bits64[NB64BLOCK - 1])>=0; }
bool Int::IsNegative() const { return (int64_t)(bits64[NB64BLOCK - 1])<0; }
bool Int::IsEven() const { return (bits[0] & 0x1) == 0; }
bool Int::IsOdd() const { return (bits[0] & 0x1) == 1; }

// -------------------
void Int::Neg() {
  unsigned char c=0;
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
void Int::SetInt32(uint32_t value) {
  CLEAR();
  bits[0]=value;
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

void Int::SetByte(int n,unsigned char byte) {
  unsigned char *bbPtr = (unsigned char *)bits;
  bbPtr[n] = byte;
}

void Int::SetDWord(int n,uint32_t b) {
  bits[n] = b;
}

void Int::SetQWord(int n, uint64_t b) {
  bits64[n] = b;
}

// -------------------------------------------
// Przesuwania w lewo i w prawo
void Int::ShiftL32Bit() {
  for(int i=NB32BLOCK-1;i>0;i--) {
    bits[i]=bits[i-1];
  }
  bits[0]=0;
}

void Int::ShiftL64Bit() {
  for (int i = NB64BLOCK-1 ; i>0; i--) {
    bits64[i] = bits64[i - 1];
  }
  bits64[0] = 0;
}

void Int::ShiftL64BitAndSub(const Int *a,int n) {
  Int b;
  int i=NB64BLOCK-1;

  for(;i>=n;i--)
    b.bits64[i] = ~a->bits64[i-n];
  for(;i>=0;i--)
    b.bits64[i] = 0xFFFFFFFFFFFFFFFFULL;

  Add(&b);
  AddOne();
}

void Int::ShiftL(uint32_t n) {
  if(n==0) return;
  if( n<64 ) {
    shiftL((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n/64;
    uint32_t nb   = n%64;
    for(uint32_t i=0;i<nb64;i++) ShiftL64Bit();
    shiftL((unsigned char)nb, bits64);
  }
}

void Int::ShiftR32Bit() {
  for(int i=0;i<NB32BLOCK-1;i++) {
    bits[i]=bits[i+1];
  }
  if(((int32_t)bits[NB32BLOCK-2])<0)
    bits[NB32BLOCK-1] = 0xFFFFFFFF;
  else
    bits[NB32BLOCK-1]=0;
}

void Int::ShiftR64Bit() {
  for (int i = 0; i<NB64BLOCK - 1; i++) {
    bits64[i] = bits64[i + 1];
  }
  if (((int64_t)bits64[NB64BLOCK - 2])<0)
    bits64[NB64BLOCK - 1] = 0xFFFFFFFFFFFFFFFF;
  else
    bits64[NB64BLOCK - 1] = 0;
}

void Int::ShiftR(uint32_t n) {
  if(n==0) return;
  if( n<64 ) {
    shiftR((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n/64;
    uint32_t nb   = n%64;
    for(uint32_t i=0;i<nb64;i++) ShiftR64Bit();
    shiftR((unsigned char)nb, bits64);
  }
}

// -------------------------------------------
void Int::SwapBit(int bitNumber) {
  uint32_t nb64 = bitNumber / 64;
  uint32_t nb = bitNumber % 64;
  uint64_t mask = 1ULL << nb;
  if(bits64[nb64] & mask ) {
    bits64[nb64] &= ~mask;
  } else {
    bits64[nb64] |= mask;
  }
}

// -------------------------------------------
void Int::Abs() {
  if (IsNegative())
    Neg();
}

// -------------------------------------------
double Int::ToDouble() const {
  double base = 1.0;
  double sum = 0;
  double pw32 = pow(2.0,32.0);
  for(int i=0;i<NB32BLOCK;i++) {
    sum += (double)(bits[i]) * base;
    base *= pw32;
  }
  return sum;
}

// -------------------------------------------
int Int::GetSize() const {
  int i=NB32BLOCK-1;
  while(i>0 && bits[i]==0) i--;
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
  int i=NB64BLOCK-1;
  while(i>=0 && t.bits64[i]==0) i--;
  if(i<0) return 0;
  return (int)((64-LZC(t.bits64[i])) + i*64);
}

// -------------------------------------------
int Int::GetLowestBit() const {
  // Zakładamy this!=0
  int b=0;
  while(((bits64[b/64] >> (b%64)) & 1) == 0) b++;
  return b;
}

// -------------------------------------------
void Int::MaskByte(int n) {
  for (int i = n; i < NB32BLOCK; i++)
    bits[i] = 0;
}

// -------------------------------------------
void Int::MatrixVecMul(Int* u,Int* v,int64_t _11,int64_t _12,int64_t _21,int64_t _22,uint64_t* cu,uint64_t* cv) {
  Int t1, t2, t3, t4;
  uint64_t c1, c2, c3, c4;
  c1 = t1.IMult(u, _11);
  c2 = t2.IMult(v, _12);
  c3 = t3.IMult(u, _21);
  c4 = t4.IMult(v, _22);
  *cu = u->AddCh(&t1, c1, &t2, c2);
  *cv = v->AddCh(&t3, c3, &t4, c4);
}

void Int::MatrixVecMul(Int* u, Int* v, int64_t _11, int64_t _12, int64_t _21, int64_t _22) {
  Int t1, t2, t3, t4;
  t1.IMult(u, _11);
  t2.IMult(v, _12);
  t3.IMult(u, _21);
  t4.IMult(v, _22);
  u->Add(&t1, &t2);
  v->Add(&t3, &t4);
}

// ------------------------------------------------

