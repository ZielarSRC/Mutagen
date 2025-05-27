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
  Point DoubleAndAdd(Point &p, Int &n);
  __attribute__((target("avx512f"))) Point DoubleAndAddAVX512(Point &p, Int &n);

  // Decoder
  bool CheckPudAddress(std::string address);
  bool DecodePrivateKey(std::string key, Int *privateKey);
  std::string GetAddress(int type, bool compressed, unsigned char *hash160);
  std::string GetBech32Address(bool compressed, Point &pubKey);
  std::string GetPrivAddressAuto(Int &privKey);

  // Address format
  static int GetPrivAddressType(std::string address);
  static int GetPublicAddressType(std::string address);
  static std::string GetPublicBase58(uint8_t *hash160, int addressType);
  static std::string GetPrivBase58(uint8_t *data, int size);
  static bool GetPrivAddr(std::string addr, uint8_t *data, int size);
  static bool IsCompressedAddress(std::string address);
  static bool IsCompressedSpark(std::string address);
  static bool IsCompressedPublic(int type);

  // Xeon Platinum 8488C optimized functions
  __attribute__((target("avx512f"))) void BatchNormalize(Point *points, int count);

  // Public curve parameters
  Int P;    // Prime for the finite field
  Int N;    // Curve order
  Int B;    // Curve parameter
  Point G;  // Generator point

 private:
  Int _Gx;  // Generator x coordinate
  Int _Gy;  // Generator y coordinate
  Int _Af;  // a coefficient (0 for secp256k1)
};

// SHA256 and RIPEMD160 functions
void SHA256(unsigned char *data, int len, unsigned char *hash);
void RIPEMD160(unsigned char *data, int len, unsigned char *hash);

// Batch processing functions
void sha256_avx512_16blocks(uint8_t **in, uint8_t **out);
void ripemd160_avx512_16blocks(const uint8_t **in, uint8_t **out);

#endif  // SECP256K1H
