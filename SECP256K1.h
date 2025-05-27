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
  Point AddJacobian(Point &p1, Point &p2);
  Point Double(Point &p);
  Point DoubleJacobian(Point &p);
  void GetHash160(int type, bool compressed, Point &pubKey, unsigned char *hash);
  void GetHash160_Batch16(int type, bool compressed, Point **pubKey, uint8_t **hash);
  std::string GetPrivAddress(bool compressed, Int &privKey);
  std::string GetPublicKeyHex(bool compressed, Point &pubKey);
  Point NextKey(Point &key);
  bool EC(Point &p);

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

  Point GJ;  // Generator Jacobian
  Point G;   // Generator

 private:
  Int _s;       // group order
  Int _p;       // prime
  Int _r;       // prime - 2
  Int _a;       // group parameter
  Int _b;       // group parameter
  Int _n;       // order of the group
  Int _nh;      // half order of the group
  Int _lambda;  // factor to convert endomorphism
  Int _Gx;      // Base point x coordinate
  Int _Gy;      // Base point y coordinate
  Point _g;     // Base point
  Point _g2;    // 2*Base point
  int _groupBSize;
  double _groupSize;
  Secp256K1 *_secp;
  std::vector<char> _lut;
  Int _2;  // Constant 2
  Int _3;  // Constant 3
};

// Deklaracje funkcji, które powodują błędy kompilacji
extern "C" {
void sha256_avx512_16blocks(uint8_t **in, uint8_t **out);
void ripemd160_avx512_16blocks(const uint8_t **in, uint8_t **out);
}

#endif  // SECP256K1H
