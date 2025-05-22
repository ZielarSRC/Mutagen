#ifndef BIGINTH
#define BIGINTH

#include <immintrin.h>
#include <inttypes.h>
#include <string>
#include <cstring>

#define BISIZE 256

#if BISIZE == 256
#define NB64BLOCK 5
#define NB32BLOCK 10
#elif BISIZE == 512
#define NB64BLOCK 9
#define NB32BLOCK 18
#else
#error Unsupported size
#endif

class Int {
 public:
  Int();
  Int(int64_t i64);
  Int(uint64_t u64);
  Int(Int *a);

  void Add(uint64_t a);
  void Add(Int *a);
  void Add(Int *a, Int *b);
  void AddOne();
  void Sub(uint64_t a);
  void Sub(Int *a);
  void Sub(Int *a, Int *b);
  void SubOne();
  void Mult(Int *a);
  uint64_t Mult(uint64_t a);
  uint64_t IMult(int64_t a);
  uint64_t Mult(Int *a, uint64_t b);
  uint64_t IMult(Int *a, int64_t b);
  void Mult(Int *a, Int *b);
  void Div(Int *a, Int *mod = NULL);
  void MultModN(Int *a, Int *b, Int *n);
  void Neg();
  void Abs();

  void ShiftR(uint32_t n);
  void ShiftR32Bit();
  void ShiftR64Bit();
  void ShiftL(uint32_t n);
  void ShiftL32Bit();
  void ShiftL64Bit();
  void SwapBit(int bitNumber);
  void Xor(const Int *a);

  bool IsGreater(Int *a);
  bool IsGreaterOrEqual(Int *a);
  bool IsLowerOrEqual(Int *a);
  bool IsLower(Int *a);
  bool IsEqual(Int *a);
  bool IsZero();
  bool IsOne();
  bool IsStrictPositive();
  bool IsPositive();
  bool IsNegative();
  bool IsEven();
  bool IsOdd();
  bool IsProbablePrime();

  double ToDouble();

  static void SetupField(Int *n, Int *R = NULL, Int *R2 = NULL, Int *R3 = NULL, Int *R4 = NULL);
  static Int *GetR();
  static Int *GetR2();
  static Int *GetR3();
  static Int *GetR4();
  static Int *GetFieldCharacteristic();
  void GCD(Int *a);
  void Mod(Int *n);
  void ModInv();
  void MontgomeryMult(Int *a, Int *b);
  void MontgomeryMult(Int *a);
  void ModAdd(Int *a);
  void ModAdd(Int *a, Int *b);
  void ModAdd(uint64_t a);
  void ModSub(Int *a);
  void ModSub(Int *a, Int *b);
  void ModSub(uint64_t a);
  void ModMul(Int *a, Int *b);
  void ModMul(Int *a);
  void ModSquare(Int *a);
  void ModCube(Int *a);
  void ModDouble();
  void ModExp(Int *e);
  void ModNeg();
  void ModSqrt();
  bool HasSqrt();
  void imm_umul_asm(const uint64_t *a, uint64_t b, uint64_t *res);

  // AVX-512 Optimized
  void MontgomeryMultAVX512(Int *a, Int *b, Int *n);
  void AddAVX512(Int *a);

  // Secp256k1 specific
  static void InitK1(Int *order);
  void ModMulK1(Int *a, Int *b);
  void ModMulK1(Int *a);
  void ModSquareK1(Int *a);
  void ModMulK1order(Int *a);
  void ModAddK1order(Int *a, Int *b);
  void ModAddK1order(Int *a);
  void ModSubK1order(Int *a);
  void ModNegK1order();
  uint32_t ModPositiveK1();

  int GetSize();
  int GetSize64();
  int GetBitLength();

  void SetInt32(uint32_t value);
  void Set(Int *a);
  void SetBase10(char *value);
  void SetBase16(char *value);
  void SetBaseN(int n, char *charset, char *value);
  void SetByte(int n, unsigned char byte);
  void SetDWord(int n, uint32_t b);
  void SetQWord(int n, uint64_t b);
  void Rand(int nbit);
  void Rand(Int *randMax);
  void Set32Bytes(unsigned char *bytes);
  void MaskByte(int n);

  uint32_t GetInt32();
  int GetBit(uint32_t n);
  unsigned char GetByte(int n);
  void Get32Bytes(unsigned char *buff);

  std::string GetBase2();
  std::string GetBase10();
  std::string GetBase16();
  std::string GetBaseN(int n, char *charset);
  std::string GetBlockStr();
  std::string GetC64Str(int nbDigit);

  static void Check();
  static bool CheckInv(Int *a);
  static Int P;

  union {
    __attribute__((aligned(64))) uint32_t bits[NB32BLOCK];
    __attribute__((aligned(64))) uint64_t bits64[NB64BLOCK];
  };

 private:
  void MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22, uint64_t *cu, uint64_t *cv);
  void MatrixVecMul(Int *u, Int *v, int64_t _11, int64_t _12, int64_t _21, int64_t _22);
  uint64_t AddCh(Int *a, uint64_t ca, Int *b, uint64_t cb);
  uint64_t AddCh(Int *a, uint64_t ca);
  uint64_t AddC(Int *a);
  void AddAndShift(Int *a, Int *b, uint64_t cH);
  void ShiftL64BitAndSub(Int *a, int n);
  uint64_t Mult(Int *a, uint32_t b);
  int GetLowestBit();
  void CLEAR();
  void CLEARFF();
  void DivStep62(Int *u, Int *v, int64_t *eta, int *pos, int64_t *uu, int64_t *uv, int64_t *vu, int64_t *vv);
};

#endif  // BIGINTH
