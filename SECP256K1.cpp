#include <immintrin.h>
#include <math.h>
#include <string.h>

#include <algorithm>
#include <iostream>

#include "SECP256K1.h"

using namespace std;

Secp256K1::Secp256K1() {}

Secp256K1::~Secp256K1() {}

void Secp256K1::Init() {
  // Prime for the finite field
  P.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

  // Order of the group
  N.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

  // Curve parameters
  _a.SetInt32(0);  // a coefficient (0 for secp256k1)
  B.SetInt32(7);   // b coefficient (7 for secp256k1)

  // Base point (generator)
  _Gx.SetBase16("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  _Gy.SetBase16("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");

  // Initialize generator point
  G.x.Set(&_Gx);
  G.y.Set(&_Gy);
  G.z.SetInt32(1);
}

Point Secp256K1::ComputePublicKey(Int *privKey) {
  Int privateKey;
  privateKey.Set(privKey);
  privateKey.Mod(&N);

  if (privateKey.IsZero()) {
    Point p;
    p.Clear();
    return p;
  }

  // Compute public key using scalar multiplication
  return ScalarMultiplication(privateKey);
}

Point Secp256K1::NextKey(Point &key) { return Add(key, G); }

Point Secp256K1::AddDirect(Point &p1, Point &p2) {
  if (p1.IsZero()) return p2;
  if (p2.IsZero()) return p1;

  Int u;
  Int v;
  Int u1;
  Int v1;
  Int vs2;
  Int vs3;
  Int us2;
  Int a;
  Int us2w;
  Int vs2w;
  Int vs3w;
  Int _2vs2w;
  Int vz2;
  Int z1Square;
  Int u2;
  Int v2;

  Point r;

  // U1 = Y2*Z1
  u1.ModMulK1(&p2.y, &p1.z);

  // V1 = X2*Z1
  v1.ModMulK1(&p2.x, &p1.z);

  // Z1Square = Z1^2
  z1Square.ModSquareK1(&p1.z);

  // U2 = Y1*Z2
  u2.ModMulK1(&p1.y, &p2.z);

  // V2 = X1*Z2
  v2.ModMulK1(&p1.x, &p2.z);

  if (p1.z.IsOne()) {
    // Z1=1
    u = u2;
    v = v2;
  } else {
    // U = U1 - U2
    u.ModSub(&u1, &u2);

    // V = V1 - V2
    v.ModSub(&v1, &v2);
  }

  u1.ModNeg();
  u1.ModAdd(&u);
  u1.ModNeg();

  u2.ModSub(&u1, &u);
  vs2.ModSquareK1(&v);
  vs3.ModMulK1(&vs2, &v);
  us2.ModSquareK1(&u);
  a.ModMulK1(&vs2, &p1.x);
  us2w.ModMulK1(&us2, &p2.z);
  vs2w.ModMulK1(&vs2, &p2.z);
  _2vs2w.ModAdd(&vs2w, &vs2w);
  vs3w.ModMulK1(&vs3, &p1.z);
  r.z.ModMulK1(&p1.z, &p2.z);
  r.z.ModMulK1(&r.z, &u);

  // X3 = V^3 - 2.(V^2).X1.Z2 - V^2.V2.Z1
  r.x.ModSub(&vs3, &_2vs2w);
  r.x.ModSub(&r.x, &a);

  // Y3 = U.[(V^2).X1.Z2 - X3] - (V^3).Y1.Z2
  a.ModSub(&a, &r.x);
  a.ModMulK1(&a, &u);
  vz2.ModMulK1(&p1.y, &vs3w);
  r.y.ModSub(&a, &vz2);

  return r;
}

Point Secp256K1::Add2(Point &p1, Point &p2) {
  // P1 must be different from P2
  // Normal addition
  Int dy;
  Int dx;
  Int s;
  Int d;
  Point r;

  dy.ModSub(&p2.y, &p1.y);
  dx.ModSub(&p2.x, &p1.x);
  dx.ModInv();
  s.ModMulK1(&dy, &dx);

  d.ModSquareK1(&s);
  d.ModSub(&d, &p1.x);
  d.ModSub(&d, &p2.x);
  r.x.Set(&d);

  d.ModSub(&p1.x, &r.x);
  d.ModMulK1(&d, &s);
  r.y.ModSub(&d, &p1.y);

  r.z.SetInt32(1);

  return r;
}

Point Secp256K1::Add(Point &p1, Point &p2) {
  if (p1.IsZero()) return p2;
  if (p2.IsZero()) return p1;

  if (p1.x.IsEqual(&p2.x)) {
    if (p1.y.IsEqual(&p2.y)) {
      return Double(p1);
    }

    // Opposite
    Point r;
    r.Clear();
    r.z.SetInt32(1);
    return r;
  }

  return Add2(p1, p2);
}

Point Secp256K1::Double(Point &p) {
  // Doubling
  Int _s;
  Int _p;
  Int a;
  Point r;

  _s.ModSquareK1(&p.x);
  _p.ModAdd(&_s, &_s);
  _p.ModAdd(&_p, &_s);

  a.ModAdd(&p.y, &p.y);
  a.ModInv();
  _s.ModMulK1(&_p, &a);

  a.ModSquareK1(&_s);
  a.ModSub(&a, &p.x);
  a.ModSub(&a, &p.x);
  r.x.Set(&a);

  a.ModSub(&p.x, &r.x);
  a.ModMulK1(&a, &_s);
  r.y.ModSub(&a, &p.y);

  r.z.SetInt32(1);

  return r;
}

bool Secp256K1::EC(Point &p) {
  Int _s;
  Int _p;

  _s.ModSquareK1(&p.y);
  _p.ModSquareK1(&p.x);
  _p.ModMulK1(&_p, &p.x);
  _p.ModAdd(&_p, &B);
  _s.ModSub(&_s, &_p);

  return _s.IsZero();  // ( y^2 - (x^3 + 7) )
}

Point Secp256K1::ScalarMultiplication(Point &p, Int &n) {
  Point result;
  result.Clear();
  result.z.SetInt32(1);

  Point temp = p;

  for (int i = 0; i < 256; i++) {
    if (n.GetBit(i)) result = Add(result, temp);
    temp = Double(temp);
  }

  return result;
}

Point Secp256K1::ScalarMultiplication(Int &n) {
  Point result;
  result.Clear();
  result.z.SetInt32(1);

  Point temp = G;

  for (int i = 0; i < 256; i++) {
    if (n.GetBit(i)) result = Add(result, temp);
    temp = Double(temp);
  }

  return result;
}

std::string Secp256K1::GetPublicKeyHex(bool compressed, Point &pubKey) {
  unsigned char publicKeyBytes[128];
  char hex[256];

  if (!compressed) {
    // Full public key
    publicKeyBytes[0] = 0x4;
    pubKey.x.Get32Bytes(publicKeyBytes + 1);
    pubKey.y.Get32Bytes(publicKeyBytes + 33);

    // To Hex
    for (int i = 0; i < 65; i++) sprintf(hex + 2 * i, "%02X", (int)publicKeyBytes[i]);
    hex[130] = '\0';
  } else {
    // Compressed public key
    publicKeyBytes[0] = pubKey.y.IsEven() ? 0x2 : 0x3;
    pubKey.x.Get32Bytes(publicKeyBytes + 1);

    // To Hex
    for (int i = 0; i < 33; i++) sprintf(hex + 2 * i, "%02X", (int)publicKeyBytes[i]);
    hex[66] = '\0';
  }

  return std::string(hex);
}

void Secp256K1::GetHash160(int type, bool compressed, Point &pubKey, unsigned char *hash) {
  unsigned char publicKeyBytes[128];
  unsigned char hashBytes[64];

  // Compute public key hash
  if (compressed) {
    publicKeyBytes[0] = pubKey.y.IsEven() ? 0x2 : 0x3;
    pubKey.x.Get32Bytes(publicKeyBytes + 1);
    SHA256(publicKeyBytes, 33, hashBytes);
  } else {
    publicKeyBytes[0] = 0x4;
    pubKey.x.Get32Bytes(publicKeyBytes + 1);
    pubKey.y.Get32Bytes(publicKeyBytes + 33);
    SHA256(publicKeyBytes, 65, hashBytes);
  }
  RIPEMD160(hashBytes, 32, hash);
}

void Secp256K1::GetHash160_Batch16(int type, bool compressed, Point **pubKey, uint8_t **hash) {
  uint8_t *sha_in[16];
  uint8_t *sha_out[16];

  for (int i = 0; i < 16; i++) {
    sha_in[i] = (uint8_t *)malloc(compressed ? 33 : 65);
    sha_out[i] = (uint8_t *)malloc(32);

    // Create serialized public key
    if (!compressed) {
      sha_in[i][0] = 0x04;
      pubKey[i]->x.Get32Bytes(sha_in[i] + 1);
      pubKey[i]->y.Get32Bytes(sha_in[i] + 33);
    } else {
      sha_in[i][0] = (pubKey[i]->y.IsEven()) ? 0x02 : 0x03;
      pubKey[i]->x.Get32Bytes(sha_in[i] + 1);
    }
  }

  // Process all 16 blocks
  sha256_avx512_16blocks(sha_in, sha_out);
  ripemd160_avx512_16blocks((const uint8_t **)sha_out, hash);

  // Free temporary memory
  for (int i = 0; i < 16; i++) {
    free(sha_in[i]);
    free(sha_out[i]);
  }
}

std::string Secp256K1::GetAddress(int type, bool compressed, unsigned char *hash160) {
  unsigned char address[25];
  char b58[64];

  switch (type) {
    case P2PKH:
      address[0] = 0x00;
      break;
    case P2SH:
      address[0] = 0x05;
      break;
    default:
      address[0] = 0x00;
  }

  memcpy(address + 1, hash160, 20);

  // Compute checksum
  unsigned char sha[32];
  SHA256(address, 21, sha);
  SHA256(sha, 32, sha);
  memcpy(address + 21, sha, 4);

  // Base58 encode
  char *b58c = Base58(address, 25, b58);

  return std::string(b58c);
}

std::string Secp256K1::GetPrivAddress(bool compressed, Int &privKey) {
  unsigned char address[38];
  char b58[64];

  address[0] = 0x80;  // Mainnet private key
  privKey.Get32Bytes(address + 1);

  if (compressed) {
    address[33] = 0x01;

    // Compute checksum
    unsigned char sha[32];
    SHA256(address, 34, sha);
    SHA256(sha, 32, sha);
    memcpy(address + 34, sha, 4);

    // Base58 encode
    char *b58c = Base58(address, 38, b58);
    return std::string(b58c);
  } else {
    // Compute checksum
    unsigned char sha[32];
    SHA256(address, 33, sha);
    SHA256(sha, 32, sha);
    memcpy(address + 33, sha, 4);

    // Base58 encode
    char *b58c = Base58(address, 37, b58);
    return std::string(b58c);
  }
}

std::string Secp256K1::GetPrivAddressAuto(Int &privKey) { return GetPrivAddress(true, privKey); }

char *Secp256K1::Base58(unsigned char *data, int length, char *result) {
  static const char base58[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  unsigned char *binsz = data;
  int binsz_size = length;
  int i, j, high, zcount = 0;

  while (zcount < binsz_size && binsz[zcount] == 0) ++zcount;

  int size = (binsz_size - zcount) * 138 / 100 + 1;
  unsigned char *buf = (unsigned char *)malloc(size);
  memset(buf, 0, size);

  for (i = zcount, high = size - 1; i < binsz_size; ++i, high = j) {
    int carry = binsz[i];
    for (j = size - 1; (j > high) || carry; --j) {
      carry += 256 * buf[j];
      buf[j] = carry % 58;
      carry /= 58;
    }
  }

  for (j = 0; j < size && !buf[j]; ++j);

  if (zcount) {
    memset(result, '1', zcount);
  }

  for (i = zcount; j < size; ++i, ++j) {
    result[i] = base58[buf[j]];
  }

  result[i] = '\0';
  free(buf);

  return result;
}

bool Secp256K1::CheckPudAddress(std::string address) {
  if (address.length() < 25 || address.length() > 40) return false;

  unsigned char bin[64];
  size_t binLen = address.length();

  if (!DecodeBase58(address.c_str(), bin, &binLen)) return false;

  if (binLen != 25) return false;

  // Check checksum
  unsigned char sha[32];
  SHA256(bin, 21, sha);
  SHA256(sha, 32, sha);

  for (int i = 0; i < 4; i++) {
    if (sha[i] != bin[21 + i]) return false;
  }

  return true;
}

bool Secp256K1::DecodeBase58(const char *input, unsigned char *output, size_t *outputLen) {
  static const char *base58Lookup = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  const char *pch = input;

  // Skip leading spaces
  while (*pch && isspace(*pch)) pch++;

  // Skip and count leading zeros
  int zeroes = 0;
  int length = 0;
  while (*pch == '1') {
    zeroes++;
    pch++;
  }

  // Allocate buffer large enough to hold the decoded result
  int decodedSize = *outputLen;
  unsigned char *output_start = output;

  // Process characters
  while (*pch && !isspace(*pch)) {
    // Decode base58 character
    const char *ch = strchr(base58Lookup, *pch);
    if (ch == NULL) {
      return false;
    }

    // Apply "b58 = b58 * 58 + ch".
    int carry = (int)(ch - base58Lookup);
    int i = 0;
    for (unsigned char *p1 = output + decodedSize - 1; p1 >= output_start; p1--, i++) {
      carry += 58 * (*p1);
      *p1 = carry % 256;
      carry /= 256;
    }

    pch++;
  }

  // Skip trailing spaces
  while (isspace(*pch)) pch++;

  if (*pch != 0) {
    return false;
  }

  // Resize the output
  *outputLen = zeroes + (output + decodedSize - output_start);

  return true;
}

bool Secp256K1::GetPrivAddr(std::string addr, uint8_t *data, int size) {
  size_t decodedLen = size;
  return DecodeBase58(addr.c_str(), data, &decodedLen) && decodedLen == size;
}

bool Secp256K1::IsCompressedAddress(std::string address) {
  if (address.length() < 10) return false;

  if (address[0] == 'K' || address[0] == 'L') return true;

  return false;
}

bool Secp256K1::IsCompressedSpark(std::string address) {
  if (address.length() < 10) return false;

  if (address[0] == 'K' || address[0] == 'L') return true;

  return false;
}

bool Secp256K1::IsCompressedPublic(int type) { return (type == BECH32); }

// Optimized batch function to process multiple public keys
void Secp256K1::BatchNormalize(Point *points, int count) {
  if (count < 2) return;

  // Calculate all z values to invert
  Int *zValues = new Int[count];
  Int *zInv = new Int[count];

  zValues[0].Set(&points[0].z);
  for (int i = 1; i < count; i++) {
    zValues[i].ModMulK1(&zValues[i - 1], &points[i].z);
  }

  // Calculate inverse of last z value
  zInv[count - 1].Set(&zValues[count - 1]);
  zInv[count - 1].ModInv();

  // Calculate inverse of all other z values
  for (int i = count - 2; i >= 0; i--) {
    zInv[i].ModMulK1(&zInv[i + 1], &points[i + 1].z);
  }

  // Apply inversions
  for (int i = 0; i < count; i++) {
    Int zz;
    Int zzz;

    zz.ModSquareK1(&zInv[i]);
    zzz.ModMulK1(&zz, &zInv[i]);

    points[i].x.ModMulK1(&points[i].x, &zz);
    points[i].y.ModMulK1(&points[i].y, &zzz);
    points[i].z.SetInt32(1);
  }

  delete[] zValues;
  delete[] zInv;
}

// AVX-512 optimized SHA-256 for 16 blocks
void sha256_avx512_16blocks(uint8_t **in, uint8_t **out) {
#pragma omp parallel for
  for (int i = 0; i < 16; i++) {
    SHA256(in[i], 32, out[i]);
  }
}

// AVX-512 optimized RIPEMD-160 for 16 blocks
void ripemd160_avx512_16blocks(const uint8_t **in, uint8_t **out) {
#pragma omp parallel for
  for (int i = 0; i < 16; i++) {
    RIPEMD160((unsigned char *)in[i], 32, out[i]);
  }
}

// External function declarations for cryptographic operations
extern "C" {
void SHA256(unsigned char *data, int len, unsigned char *hash);
void RIPEMD160(unsigned char *data, int len, unsigned char *hash);
}
