#ifndef SECP256K1_H
#define SECP256K1_H

#include <cstdint>
#include <string>
#include <vector>

#include "Int.h"
#include "Point.h"

enum AddressType { P2PKH = 0, P2SH = 1, BECH32 = 2 };

class Secp256K1 {
 public:
  Secp256K1();
  ~Secp256K1();

  void Init();
  Point ComputePublicKey(Int *privKey);
  Point NextKey(Point &key);

  void GetHash160(int type, bool compressed, Point &pubKey, unsigned char *hash);
  void GetHash160_Batch16(int type, bool compressed, Point *pubkeys[16], uint8_t *hashes[16]);

  std::string GetAddress(int type, bool compressed, Point &pubKey);
  std::string GetAddress(int type, bool compressed, unsigned char *hash160);
  std::string GetPrivAddress(bool compressed, Int &privKey);
  std::string GetPublicKeyHex(bool compressed, Point &p);
  Point ParsePublicKeyHex(std::string str, bool &isCompressed);
  bool CheckPudAddress(std::string address);
  static Int DecodePrivateKey(char *key, bool *compressed);

  Point AddDirect(Point &p1, Point &p2);
  Point DoubleDirect(Point &p);
  Point Add(Point &p1, Point &p2);
  Point Double(Point &p);
  Point Add2(Point &p1, Point &p2);
  bool EC(Point &p);
  Int GetY(Int x, bool isEven);

  // Public members - zgodnie z działającą wersją
  Point G;    // Generator point
  Int order;  // Curve order

 private:
  void SerializePublicKey(Point &pubKey, bool compressed, uint8_t *out33);
  uint8_t GetByte(std::string &str, int idx);

  // Generator table - wyrównany do cache line
  alignas(64) Point GTable[256 * 32];
};

#endif  // SECP256K1_H
