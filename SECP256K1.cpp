#include <math.h>
#include <string.h>

#include <algorithm>
#include <iostream>

#include "SECP256K1.h"
#include "ripemd160_avx512.h"
#include "sha256_avx512.h"

using namespace std;

Secp256K1::Secp256K1() {}

void Secp256K1::Init() {
  // Prime for the finite field
  _p.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

  // Order of the group
  _s.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

  // Half order of the group
  //_halfS.SetBase16("7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0");

  _a.SetInt32(0);
  _b.SetInt32(7);

  // Base point (generator)
  _Gx.SetBase16("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  _Gy.SetBase16("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");

  // Compute generator table
  Point G(_Gx, _Gy);
  G.Init(&_p, &_s);

  _groupBSize = 256;
  _groupSize = pow(2, _groupBSize);

  // Compute Generator table
  Point g = G;
  G.Set(g);
  _g[0] = g;
  g = g.DoubleDirect();
  G.Set(g);
  _g[1] = g;
  for (int i = 2; i < 32; i++) {
    g = g.DoubleDirect();
    G.Set(g);
    _g[i] = g;
  }

  // For Endomorphism
  // _beta.SetBase16("7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE");
  // _lambda.SetBase16("5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72");

  _two.SetInt32(2);
  _two = _two.ModInv(&_s);
}

Secp256K1::~Secp256K1() {}

Point Secp256K1::ComputePublicKey(Int* privKey) {
  int i = 0;
  uint8_t b;
  Point Q;
  Int P;
  P.Set(privKey);
  P.Mod(&_s);

  // Search first significant byte
  for (i = 0; i < 32; i++) {
    b = P.GetByte(i);
    if (b) break;
  }
  if (i == 32) {
    // Null private key, Neutral element
    Q.Clear();
    Q.z.SetInt32(1);
    return Q;
  }

  // Compute pub key
  int bLength = (256 - i * 8);
  Q = DoubleAndAdd(&_g[0], P, i, bLength);

  Q.Normalize();
  return Q;
}

Point Secp256K1::NextKey(Point& key) {
  // Input key must be normalized
  Point r = AddDirect(key, _g[0]);
  r.Normalize();
  return r;
}

// P = k*A
Point Secp256K1::ScalarMultiplication(Point& A, Int* k) {
  Point R;
  R.Clear();
  R.z.SetInt32(1);
  return DoubleAndAdd(&A, *k, 0, 256);
}

// P = k*G
Point Secp256K1::ScalarMultiplication(Int* k) {
  Point R;
  R.Clear();
  R.z.SetInt32(1);
  return DoubleAndAdd(&_g[0], *k, 0, 256);
}

// Double and Add (fast) - Miller Rabin
Point Secp256K1::DoubleAndAdd(Point* P, Int& n, int from, int length) {
  Point R;
  R.Clear();
  R.z.SetInt32(1);
  int i;
  int naf_length;
  signed char* naf = n.GetNAF(&naf_length, length);
  int nbBit = naf_length;
  for (i = nbBit - 1; i >= from; i--) {
    R = R.DoubleDirect();
    if (naf[i] > 0) {
      R = AddDirect(R, P[0]);
    } else if (naf[i] < 0) {
      Point& ng = P[0].Neg();
      R = AddDirect(R, ng);
    }
  }
  free(naf);
  return R;
}

void Secp256K1::GetPublicKeyHex(bool compressed, Point& pubKey, char* dst) {
  unsigned char publicKeyBytes[128];

  if (!compressed) {
    // Full public key
    publicKeyBytes[0] = 0x4;
    pubKey.x.Get32Bytes(publicKeyBytes + 1);
    pubKey.y.Get32Bytes(publicKeyBytes + 33);
    printf("PubKeyHex: %s\n", publicKeyBytes);

    // To Hex
    for (int i = 0; i < 65; i++) sprintf(dst + 2 * i, "%02X", (int)publicKeyBytes[i]);

  } else {
    // Compressed public key
    publicKeyBytes[0] = pubKey.y.IsEven() ? 0x2 : 0x3;
    pubKey.x.Get32Bytes(publicKeyBytes + 1);

    // To Hex
    for (int i = 0; i < 33; i++) sprintf(dst + 2 * i, "%02X", (int)publicKeyBytes[i]);
  }
}

Point Secp256K1::AddDirect(Point& p1, Point& p2) {
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

  u1.ModNeg(&u);
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

Point Secp256K1::Add2(Point& p1, Point& p2) {
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

Point Secp256K1::Add(Point& p1, Point& p2) {
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

Point Secp256K1::Double(Point& p) {
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

bool Secp256K1::EC(Point& p) {
  Int _s;
  Int _p;

  _s.ModSquareK1(&p.y);
  _p.ModSquareK1(&p.x);
  _p.ModMulK1(&_p, &p.x);
  _p.ModAdd(&_p, &_b);
  _s.ModSub(&_s, &_p);

  return _s.IsZero();  // ( y^2 - (x^3 + 7) )
}

void Secp256K1::GetHash160(int type, bool compressed, Point& pubKey, unsigned char* hash) {
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

// Compute SHA-256 for 16 blocks in parallel using AVX-512
void sha256_avx512_16blocks(uint8_t** in, uint8_t** out) {
  for (int i = 0; i < 16; i++) {
    SHA256(in[i], 32, out[i]);
  }
}

// Compute RIPEMD-160 for 16 blocks in parallel using AVX-512
void ripemd160_avx512_16blocks(const uint8_t** in, uint8_t** out) {
  for (int i = 0; i < 16; i++) {
    RIPEMD160(in[i], 32, out[i]);
  }
}

void Secp256K1::GetHash160_Batch16(int type, bool compressed, Point** pubKey, uint8_t** hash) {
  uint8_t* sha[16];
  uint8_t* sha_in[16];
  uint8_t* sha_out[16];

  for (int i = 0; i < 16; i++) {
    sha[i] = (uint8_t*)malloc(32);
    sha_in[i] = (uint8_t*)malloc(compressed ? 33 : 65);
    sha_out[i] = (uint8_t*)malloc(32);

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
  ripemd160_avx512_16blocks((const uint8_t**)sha_out, hash);

  // Free temporary memory
  for (int i = 0; i < 16; i++) {
    free(sha[i]);
    free(sha_in[i]);
    free(sha_out[i]);
  }
}

std::string Secp256K1::GetAddress(int type, bool compressed, unsigned char* hash160) {
  uint8_t* in[16];
  uint8_t* out[16];

  for (int i = 0; i < 16; i++) {
    in[i] = (uint8_t*)malloc(32);
    out[i] = (uint8_t*)malloc(32);
  }

  // Setup input for SHA-256
  uint8_t tmp[25];

  switch (type) {
    case P2PKH:
      tmp[0] = 0x00;
      break;
    case P2SH:
      tmp[0] = 0x05;
      break;
    default:
      tmp[0] = 0x00;
  }

  memcpy(tmp + 1, hash160, 20);

  // Calculate checksum using SHA-256
  for (int i = 0; i < 16; i++) {
    memcpy(in[i], tmp, 21);
  }

  sha256_avx512_16blocks(in, out);
  sha256_avx512_16blocks(out, in);

  // Build the base58Check address
  memcpy(tmp + 21, in[0], 4);

  // Encode base58
  char result[64];
  char* str = Base58Encode(tmp, 25, result);

  // Free allocated memory
  for (int i = 0; i < 16; i++) {
    free(in[i]);
    free(out[i]);
  }

  return std::string(str);
}

bool Secp256K1::CheckPudAddress(std::string address) {
  uint8_t* in[16];
  uint8_t* out[16];

  for (int i = 0; i < 16; i++) {
    in[i] = (uint8_t*)malloc(32);
    out[i] = (uint8_t*)malloc(32);
  }

  // Decode base58
  uint8_t pubKeyBytes[128];
  int pubKeyLen = Base58Decode(address.c_str(), pubKeyBytes);

  if (pubKeyLen != 25) {
    for (int i = 0; i < 16; i++) {
      free(in[i]);
      free(out[i]);
    }
    return false;
  }

  // Check checksum
  for (int i = 0; i < 16; i++) {
    memcpy(in[i], pubKeyBytes, 21);
  }

  sha256_avx512_16blocks(in, out);
  sha256_avx512_16blocks(out, in);

  // Compare checksum
  bool isValid = true;
  for (int i = 0; i < 4; i++) {
    if (in[0][i] != pubKeyBytes[21 + i]) {
      isValid = false;
      break;
    }
  }

  // Free allocated memory
  for (int i = 0; i < 16; i++) {
    free(in[i]);
    free(out[i]);
  }

  return isValid;
}

std::string Secp256K1::GetPrivAddress(bool compressed, Int& privKey) {
  uint8_t* in[16];
  uint8_t* out[16];

  for (int i = 0; i < 16; i++) {
    in[i] = (uint8_t*)malloc(32);
    out[i] = (uint8_t*)malloc(32);
  }

  // Encode private key
  uint8_t privKeyBytes[38];
  privKeyBytes[0] = 0x80;
  privKey.Get32Bytes(privKeyBytes + 1);

  if (compressed) {
    // Compressed private key
    privKeyBytes[33] = 0x01;

    for (int i = 0; i < 16; i++) {
      memcpy(in[i], privKeyBytes, 34);
    }

    sha256_avx512_16blocks(in, out);
    sha256_avx512_16blocks(out, in);

    // Add checksum
    memcpy(privKeyBytes + 34, in[0], 4);

    // Encode base58
    char result[64];
    char* str = Base58Encode(privKeyBytes, 38, result);

    // Free allocated memory
    for (int i = 0; i < 16; i++) {
      free(in[i]);
      free(out[i]);
    }

    return std::string(str);
  } else {
    // Uncompressed private key
    for (int i = 0; i < 16; i++) {
      memcpy(in[i], privKeyBytes, 33);
    }

    sha256_avx512_16blocks(in, out);
    sha256_avx512_16blocks(out, in);

    // Add checksum
    memcpy(privKeyBytes + 33, in[0], 4);

    // Encode base58
    char result[64];
    char* str = Base58Encode(privKeyBytes, 37, result);

    // Free allocated memory
    for (int i = 0; i < 16; i++) {
      free(in[i]);
      free(out[i]);
    }

    return std::string(str);
  }
}

std::string Secp256K1::GetPrivAddressAuto(Int& privKey) { return GetPrivAddress(true, privKey); }

bool Secp256K1::CheckPoint(Point& p) {
  // Check that point is on the curve
  return EC(p);
}

char* Secp256K1::Base58Encode(const unsigned char* data, int length, char* result) {
  static const char base58Alphabet[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  Int bn58(58);
  Int bn0(0);
  Int bn;

  bn.SetBytes(data, length);

  // Convert to base58
  std::string str;
  while (bn.IsGreaterOrEqual(&bn58)) {
    Int r;
    bn.Div(&bn58, &bn, &r);
    str.insert(0, 1, base58Alphabet[r.GetInt32()]);
  }

  if (bn.IsGreaterOrEqual(&bn0)) {
    str.insert(0, 1, base58Alphabet[bn.GetInt32()]);
  }

  // Leading zeros
  for (int i = 0; i < length && data[i] == 0; i++) {
    str.insert(0, 1, '1');
  }

  strcpy(result, str.c_str());
  return result;
}

int Secp256K1::Base58Decode(const char* address, unsigned char* result) {
  static const char base58Alphabet[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

  Int bn58(58);
  Int bn;
  Int mul;

  // Decode base58
  for (int i = 0; address[i]; i++) {
    const char* c = strchr(base58Alphabet, address[i]);
    if (c == NULL) return 0;

    int digit = c - base58Alphabet;
    mul.Set(&bn);
    mul.Mult(&bn58);
    bn.Set(&mul);
    bn.Add(digit);
  }

  // Count leading zeros
  int leadingZeros = 0;
  for (int i = 0; address[i] == '1'; i++) {
    leadingZeros++;
  }

  // Convert to bytes
  int size = bn.GetBitLength() / 8 + 1;
  bn.Get32Bytes(result + leadingZeros);

  return leadingZeros + size;
}
