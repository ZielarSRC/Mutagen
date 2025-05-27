#ifndef SECP256K1H
#define SECP256K1H

#include <string>
#include <vector>

#include "Int.h"
#include "IntGroup.h"
#include "Point.h"

// Address types
#define P2PKH 0
#define P2SH 1
#define BECH32 2

class Secp256K1 {
 public:
  Secp256K1();
  ~Secp256K1();
  void Init();
  Point ComputePublicKey(Int *privKey);
  Point AddDirect(Point &p1, Point &p2);
  Point Add(Point &p1, Point &p2);
  Point Add2(Point &p1, Point &p2);
  Point Double(Point &p);
  void GetHash160(int type, bool compressed, Point &pubKey, unsigned char *hash);
  void GetHash160_Batch16(int type, bool compressed, Point **pubKey, uint8_t **hash);
  std::string GetPrivAddress(bool compressed, Int &privKey);
  std::string GetPublicKeyHex(bool compressed, Point &pubKey);
  Point NextKey(Point &key);
  bool EC(Point &p);
  Point ScalarMultiplication(Point &p, Int &n);
  Point ScalarMultiplication(Int &n);

  // Batch operations
  void BatchNormalize(Point *points, int count);

  // Address handling
  std::string GetAddress(int type, bool compressed, unsigned char *hash160);
  std::string GetPrivAddressAuto(Int &privKey);
  bool CheckPudAddress(std::string address);

  // Base58 encoding/decoding functions
  char *Base58(unsigned char *data, int length, char *result);
  bool DecodeBase58(const char *input, unsigned char *output, size_t *outputLen);
  static bool GetPrivAddr(std::string addr, uint8_t *data, int size);

  // Utility functions
  static bool IsCompressedAddress(std::string address);
  static bool IsCompressedSpark(std::string address);
  static bool IsCompressedPublic(int type);

  // Public curve parameters
  Int P;    // Prime for the finite field
  Int N;    // Curve order
  Int B;    // Curve parameter
  Point G;  // Generator point
  Int _a;   // a coefficient (0 for secp256k1)

 private:
  Int _Gx;  // Generator x coordinate
  Int _Gy;  // Generator y coordinate
};

// Hash functions
void SHA256(unsigned char *data, int len, unsigned char *hash);
void RIPEMD160(unsigned char *data, int len, unsigned char *hash);

// Optimized hash functions for AVX-512
void sha256_avx512_16blocks(uint8_t **in, uint8_t **out);
void ripemd160_avx512_16blocks(const uint8_t **in, uint8_t **out);

#endif  // SECP256K1H
