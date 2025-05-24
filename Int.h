#ifndef BIGINTH
#define BIGINTH

#include <string>
#include <inttypes.h>
#include <immintrin.h>
#include <stdint.h>
#include <cstddef>

// We need 1 extra block for Knuth div algorithm , Montgomery multiplication and ModInv
#define BISIZE 256

#if BISIZE==256
  #define NB64BLOCK 5
  #define NB32BLOCK 10
#elif BISIZE==512
  #define NB64BLOCK 9
  #define NB32BLOCK 18
#else
  #error Unsuported size
#endif

class Int {
public:
  // Constructors
  Int();
  Int(int64_t i64);
  Int(uint64_t u64);
  Int(const Int *a);

  // Arithmetic Operations
  void Add(uint64_t a);
  void Add(const Int *a);
  void Add(const Int *a, const Int *b);
  void AddOne();
  void Sub(uint64_t a);
  void Sub(const Int *a);
  void Sub(const Int *a, const Int *b);
  void SubOne();
  void Mult(const Int *a);
  uint64_t Mult(uint64_t a);
  uint64_t IMult(int64_t a);
  uint64_t Mult(const Int *a, uint64_t b);
  uint64_t IMult(const Int *a, int64_t b);
  void Mult(const Int *a, const Int *b);
  void Div(const Int *a, Int *mod = nullptr);
  void MultModN(const Int *a, const Int *b, const Int *n);
  void Neg();
  void Abs();

  // Bitwise Operations
  void ShiftR(uint32_t n);
  void ShiftR32Bit();
  void ShiftR64Bit();
  void ShiftL(uint32_t n);
  void ShiftL32Bit();
  void ShiftL64Bit();
  void SwapBit(int bitNumber);
  void Xor(const Int *a);

  // Comparison Operations
  bool IsGreater(const Int *a) const;
  bool IsGreaterOrEqual(const Int *a) const;
  bool IsLowerOrEqual(const Int *a) const;
  bool IsLower(const Int *a) const;
  bool IsEqual(const Int *a) const;
  bool IsZero() const;
  bool IsOne() const;
  bool IsStrictPositive() const;
  bool IsPositive() const;
  bool IsNegative() const;
  bool IsEven() const;
  bool IsOdd() const;

  // Conversion
  double ToDouble() const;

  // Modular Arithmetic
  static void SetupField(const Int *n, Int *R = nullptr, Int *R2 = nullptr, Int *R3 = nullptr, Int *R4 = nullptr);
  static Int *GetR();
  static Int *GetR2();
  static Int *GetR3();
  static Int *GetR4();
  static Int* GetFieldCharacteristic();

  void GCD(const Int *a);
  void Mod(const Int *n);
  void ModInv();
  void MontgomeryMult(const Int *a, const Int *b);
  void MontgomeryMult(const Int *a);
  void ModAdd(const Int *a);
  void ModAdd(const Int *a, const Int *b);
  void ModAdd(uint64_t a);
  void ModSub(const Int *a);
  void ModSub(const Int *a, const Int *b);
  void ModSub(uint64_t a);
  void ModMul(const Int *a, const Int *b);
  void ModMul(const Int *a);
  void ModSquare(const Int *a);
  void ModCube(const Int *a);
  void ModDouble();
  void ModExp(const Int *e);
  void ModNeg();
  void ModSqrt();
  bool HasSqrt();

  // Secp256k1 Specific
  static void InitK1(const Int* order);
  void ModMulK1(const Int *a, const Int *b);
  void ModMulK1(const Int *a);
  void ModSquareK1(const Int *a);
  void ModMulK1order(const Int *a);
  void ModAddK1order(const Int *a, const Int *b);
  void ModAddK1order(const Int *a);
  void ModSubK1order(const Int *a);
  void ModNegK1order();
  uint32_t ModPositiveK1();

  // Size Information
  int GetSize() const;       
  int GetSize64() const;     
  int GetBitLength() const;  

  // Setters
  void SetInt32(uint32_t value);
  void Set(const Int *a);
  void SetBase10(const char *value);
  void SetBase16(const char *value);
  void SetBaseN(int base, const char *charset, const char *value);
  void SetByte(int n, unsigned char byte);
  void SetDWord(int n, uint32_t b);
  void SetQWord(int n, uint64_t b);
  void Rand(int nbit);
  void Rand(const Int *randMax);
  void Set32Bytes(const uint8_t *bytes);
  void MaskByte(int n);

  // Getters
  uint32_t GetInt32() const;
  int GetBit(uint32_t n) const;
  unsigned char GetByte(int n) const;
  void Get32Bytes(uint8_t *buff) const;

  // String Conversions
  std::string GetBase2() const;
  std::string GetBase10() const;
  std::string GetBase16() const;
  std::string GetBaseN(int n, const char *charset) const;
  std::string GetBlockStr() const;
  std::string GetC64Str(int nbDigit) const;

  // Validation
  static void Check();
  static bool CheckInv(const Int *a);
  static Int P;

  // Data Storage
  union {
    uint32_t bits[NB32BLOCK];
    uint64_t bits64[NB64BLOCK];
    __m512i vec512[2];  // AVX-512 optimized storage
  };

private:
  // Internal Helper Methods
  void MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22, uint64_t *cu, uint64_t *cv);
  void MatrixVecMul(Int* u, Int* v, int64_t _11, int64_t _12, int64_t _21, int64_t _22);
  uint64_t AddCh(const Int *a, uint64_t ca, const Int *b, uint64_t cb);
  uint64_t AddCh(const Int *a, uint64_t ca);
  uint64_t AddC(const Int *a);
  void AddAndShift(const Int *a, const Int *b, uint64_t cH);
  void ShiftL64BitAndSub(const Int *a, int n);
  int  GetLowestBit() const;
  void CLEAR();
  void CLEARFF();
  void DivStep62(const Int* u, const Int* v, int64_t* eta, int *pos, int64_t* uu, int64_t* uv, int64_t* vu, int64_t* vv);
  static const Int SECP_P;

  // AVX-512 Optimized Operations
  __m512i _mm512_chain_add_epi64(__m512i a, __m512i b);
  __m512i _mm512_chain_sub_epi64(__m512i a, __m512i b);
  void _mm512_montgomery_reduce(__m512i& product, const __m512i& mod, __m512i& inv);
};

// Inline AVX-512 Optimized Implementations
#ifndef WIN64

// AVX-512 optimized Montgomery multiplication
inline __m512i Int::_mm512_chain_add_epi64(__m512i a, __m512i b) {
    __m512i sum = _mm512_add_epi64(a, b);
    __m512i carry = _mm512_srli_epi64(_mm512_sub_epi64(sum, a), 63);
    return _mm512_add_epi64(sum, carry);
}

inline __m512i Int::_mm512_chain_sub_epi64(__m512i a, __m512i b) {
    __m512i diff = _mm512_sub_epi64(a, b);
    __m512i borrow = _mm512_srli_epi64(_mm512_add_epi64(diff, b), 63);
    return _mm512_sub_epi64(diff, borrow);
}

inline void Int::_mm512_montgomery_reduce(__m512i& product, const __m512i& mod, __m512i& inv) {
    __m512i q = _mm512_mullo_epi64(product, inv);
    __m512i t = _mm512_add_epi64(product, _mm512_mullo_epi64(q, mod));
    product = _mm512_srli_epi64(t, 64);
}

// Missing intrinsics for non-Windows
inline uint64_t _umul128(uint64_t a, uint64_t b, uint64_t *h) {
#if defined(__BMI2__)
    uint64_t rlo, rhi;
    __asm__ (
        "mulx %[B], %[LO], %[HI]"   
        : [LO]"=r"(rlo), [HI]"=r"(rhi)
        : "d"(a), [B]"r"(b)
        : "cc"  
    );
    *h = rhi;
    return rlo;
#else
    uint64_t rhi, rlo;
    __asm__ (
        "mulq %[B]"
        : "=d"(rhi), "=a"(rlo)
        : "a"(a), [B]"r"(b)
        : "cc"
    );
    *h = rhi;
    return rlo;
#endif
}

inline int64_t _mul128(int64_t a, int64_t b, int64_t *h) {
    uint64_t rhi, rlo;
    __asm__( "imulq %[b];" :"=d"(rhi),"=a"(rlo) :"a"(a),[b]"rm"(b));
    *h = (int64_t)rhi;
    return (int64_t)rlo;
}

static inline uint64_t _udiv128(uint64_t hi, uint64_t lo, uint64_t d, uint64_t *r) {
    uint64_t q, rem;
    asm (
        "divq %4"
        : "=d"(rem), "=a"(q)
        : "a"(lo), "d"(hi), "r"(d)
        : "cc"
    );
    *r = rem;
    return q;
}

static inline uint64_t my_rdtsc() {
    uint32_t h, l;
    __asm__( "rdtsc;" :"=d"(h),"=a"(l));
    return (uint64_t)h << 32 | (uint64_t)l;
}

#define __shiftright128(a,b,n) ((a)>>(n))|((b)<<(64-(n)))
#define __shiftleft128(a,b,n) ((b)<<(n))|((a)>>(64-(n)))

#define _subborrow_u64(a,b,c,d) __builtin_ia32_sbb_u64(a,b,c,(long long unsigned int*)d)
#define _addcarry_u64(a,b,c,d) __builtin_ia32_addcarryx_u64(a,b,c,(long long unsigned int*)d)
#define _byteswap_uint64 __builtin_bswap64
#define LZC(x) __builtin_clzll(x)
#define TZC(x) __builtin_ctzll(x)

#else

// Windows intrinsics
#include <intrin.h>
#define TZC(x) _tzcnt_u64(x)
#define LZC(x) _lzcnt_u64(x)

#endif

// Optimized 512-bit multiplication using AVX-512
inline void avx512_mul(const uint64_t *a, const uint64_t *b, uint64_t *res) {
    __m512i va = _mm512_loadu_si512(a);
    __m512i vb = _mm512_loadu_si512(b);
    
    // Perform 64x64->128 bit multiplication
    __m512i lo = _mm512_mullo_epi64(va, vb);
    __m512i hi = _mm512_mulhi_epu64(va, vb);
    
    // Store results
    _mm512_storeu_si512(res, lo);
    _mm512_storeu_si512(res + 8, hi);
}

// Optimized modular addition
inline void avx512_mod_add(uint64_t *a, uint64_t *b, uint64_t *mod, uint64_t *res) {
    __m512i va = _mm512_loadu_si512(a);
    __m512i vb = _mm512_loadu_si512(b);
    __m512i vmod = _mm512_loadu_si512(mod);
    
    __m512i sum = _mm512_add_epi64(va, vb);
    __m512i cmp = _mm512_cmpgt_epi64(sum, vmod);
    __m512i res_vec = _mm512_mask_sub_epi64(sum, cmp, sum, vmod);
    
    _mm512_storeu_si512(res, res_vec);
}

// Optimized shift operations
inline void avx512_shift_left(uint64_t *a, unsigned n, uint64_t *res) {
    __m512i vec = _mm512_loadu_si512(a);
    __m512i shifted = _mm512_slli_epi64(vec, n);
    _mm512_storeu_si512(res, shifted);
}

inline void avx512_shift_right(uint64_t *a, unsigned n, uint64_t *res) {
    __m512i vec = _mm512_loadu_si512(a);
    __m512i shifted = _mm512_srli_epi64(vec, n);
    _mm512_storeu_si512(res, shifted);
}

// Helper macros for AVX-512 operations
#define AVX512_LOAD(a) _mm512_loadu_si512((__m512i*)(a))
#define AVX512_STORE(a, v) _mm512_storeu_si512((__m512i*)(a), v)
#define AVX512_ADD(a, b) _mm512_add_epi64(a, b)
#define AVX512_SUB(a, b) _mm512_sub_epi64(a, b)
#define AVX512_MUL_LO(a, b) _mm512_mullo_epi64(a, b)
#define AVX512_MUL_HI(a, b) _mm512_mulhi_epu64(a, b)
#define AVX512_CMPGT(a, b) _mm512_cmpgt_epi64(a, b)
#define AVX512_MASK_SUB(mask, a, b) _mm512_mask_sub_epi64(a, mask, a, b)

#endif // BIGINTH
