#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "SECP256K1.h"
#include "ripemd160_avx512.h"
#include "sha256_avx512.h"

static Int SECP256K1_P([] {
  Int p;
  p.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
  return p;
}());

void Secp256K1::SerializePublicKey(Point& pubKey, bool compressed, uint8_t* out33) {
  Int x(pubKey.x), y(pubKey.y);
  x.Mod(&SECP256K1_P);
  y.Mod(&SECP256K1_P);
  if (compressed) {
    out33[0] = y.IsEven() ? 0x02 : 0x03;
    x.Get32Bytes(out33 + 1);
  } else {
    out33[0] = 0x04;
    x.Get32Bytes(out33 + 1);
    y.Get32Bytes(out33 + 33);
  }
}

Secp256K1::Secp256K1() {}
Secp256K1::~Secp256K1() {}

void Secp256K1::Init() {
  std::cout << "🔧 Inicjalizacja SECP256K1 - krok 1: tworzenie obiektu..." << std::endl;

  Int P;
  P.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
  Int::SetupField(&P);

  G.x.SetBase16("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  G.y.SetBase16("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
  G.z.SetInt32(1);
  order.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

  std::cout << "🔧 Inicjalizacja SECP256K1 - krok 2: wywołanie Init()..." << std::endl;

  Int::InitK1(&order);

  std::cout << "🔧 Inicjalizacja SECP256K1 - krok 3: budowanie tabeli generatora..." << std::endl;

  // Timeout mechanism
  auto start_time = std::chrono::high_resolution_clock::now();
  const int TIMEOUT_SECONDS = 60;

  // Compute Generator table
  Point N(G);
  for (int i = 0; i < 32; i++) {
    N.Reduce();
    GTable[i * 256] = N;
    N = DoubleDirect(N);
    for (int j = 1; j < 255; j++) {
      N.Reduce();
      GTable[i * 256 + j] = N;
      N = AddDirect(N, GTable[i * 256]);
    }
    N.Reduce();
    GTable[i * 256 + 255] = N;
  }

  // --------- AVX-512 batch hash160 helpers ---------

  void Secp256K1::GetHash160_Batch16(int type, bool compressed, Point* pubkeys[16],
                                     uint8_t* hashes[16]) {
    alignas(64) uint8_t pubkey_ser[16][33];
    const uint8_t* in[16];
    uint8_t* out[16];
    for (int i = 0; i < 16; ++i) {
      SerializePublicKey(*pubkeys[i], compressed, pubkey_ser[i]);
      in[i] = pubkey_ser[i];
      out[i] = hashes[i];
    }
    alignas(64) uint8_t sha[16][32];
    const uint8_t* sha_in[16];
    uint8_t* sha_out[16];
    for (int i = 0; i < 16; ++i) {
      sha_in[i] = in[i];
      sha_out[i] = sha[i];
    }
    sha256_avx512_16blocks(sha_in, sha_out);
    ripemd160_avx512_16blocks((const uint8_t**)sha, out);
  }

  void Secp256K1::GetHash160_Batch8(int type, bool compressed, Point* pubkeys[8],
                                    uint8_t* hashes[8]) {
    alignas(64) uint8_t pubkey_ser[16][33] = {};
    const uint8_t* in[16] = {};
    uint8_t* out[16] = {};
    for (int i = 0; i < 8; ++i) {
      SerializePublicKey(*pubkeys[i], compressed, pubkey_ser[i]);
      in[i] = pubkey_ser[i];
      out[i] = hashes[i];
    }
    for (int i = 8; i < 16; ++i) {
      in[i] = pubkey_ser[i];
      out[i] = pubkey_ser[i];
    }
    alignas(64) uint8_t sha[16][32];
    const uint8_t* sha_in[16];
    uint8_t* sha_out[16];
    for (int i = 0; i < 16; ++i) {
      sha_in[i] = in[i];
      sha_out[i] = sha[i];
    }
    sha256_avx512_16blocks(sha_in, sha_out);
    ripemd160_avx512_16blocks((const uint8_t**)sha, out);
  }

  void Secp256K1::GetHash160_Batch4(int type, bool compressed, Point* pubkeys[4],
                                    uint8_t* hashes[4]) {
    alignas(64) uint8_t pubkey_ser[16][33] = {};
    const uint8_t* in[16] = {};
    uint8_t* out[16] = {};
    for (int i = 0; i < 4; ++i) {
      SerializePublicKey(*pubkeys[i], compressed, pubkey_ser[i]);
      in[i] = pubkey_ser[i];
      out[i] = hashes[i];
    }
    for (int i = 4; i < 16; ++i) {
      in[i] = pubkey_ser[i];
      out[i] = pubkey_ser[i];
    }
    alignas(64) uint8_t sha[16][32];
    const uint8_t* sha_in[16];
    uint8_t* sha_out[16];
    for (int i = 0; i < 16; ++i) {
      sha_in[i] = in[i];
      sha_out[i] = sha[i];
    }
    sha256_avx512_16blocks(sha_in, sha_out);
    ripemd160_avx512_16blocks((const uint8_t**)sha, out);
  }

  void Secp256K1::GetHash160(int type, bool compressed, Point& pubKey, unsigned char* hash) {
    alignas(64) uint8_t pubkey_ser[16][33] = {};
    const uint8_t* in[16] = {};
    uint8_t* out[16] = {};
    SerializePublicKey(pubKey, compressed, pubkey_ser[0]);
    in[0] = pubkey_ser[0];
    out[0] = hash;
    for (int i = 1; i < 16; ++i) {
      in[i] = pubkey_ser[i];
      out[i] = pubkey_ser[i];
    }
    alignas(64) uint8_t sha[16][32];
    const uint8_t* sha_in[16];
    uint8_t* sha_out[16];
    for (int i = 0; i < 16; ++i) {
      sha_in[i] = in[i];
      sha_out[i] = sha[i];
    }
    sha256_avx512_16blocks(sha_in, sha_out);
    ripemd160_avx512_16blocks((const uint8_t**)sha, out);
  }

  std::string Secp256K1::GetAddress(int type, bool compressed, Point& pubKey) {
    unsigned char hash160[20];
    GetHash160(type, compressed, pubKey, hash160);
    return GetAddress(type, compressed, hash160);
  }

  std::string Secp256K1::GetAddress(int type, bool compressed, unsigned char* hash160) {
    unsigned char checksum[32];
    unsigned char address[25];

    if (type == P2PKH) {
      address[0] = 0x00;
    } else if (type == P2SH) {
      address[0] = 0x05;
    }

    memcpy(address + 1, hash160, 20);

    alignas(64) uint8_t inblock[16][32] = {};
    const uint8_t* in[16] = {};
    uint8_t* out[16] = {};
    memcpy(inblock[0], address, 21);
    in[0] = inblock[0];
    out[0] = inblock[1];
    sha256_avx512_16blocks(in, out);

    in[0] = inblock[1];
    out[0] = inblock[2];
    sha256_avx512_16blocks(in, out);

    memcpy(address + 21, inblock[2], 4);

    std::string result;
    const char* base58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    std::vector<uint8_t> data(address, address + 25);
    std::vector<uint8_t> encoded;

    for (size_t i = 0; i < data.size(); ++i) {
      int carry = data[i];
      for (size_t j = 0; j < encoded.size(); ++j) {
        carry += encoded[j] * 256;
        encoded[j] = carry % 58;
        carry /= 58;
      }
      while (carry > 0) {
        encoded.push_back(carry % 58);
        carry /= 58;
      }
    }

    for (size_t i = 0; i < data.size() && data[i] == 0; ++i) {
      result += base58[0];
    }

    for (int i = encoded.size() - 1; i >= 0; --i) {
      result += base58[encoded[i]];
    }

    return result;
  }

  std::vector<std::string> Secp256K1::GetAddress(int type, bool compressed, unsigned char* h1,
                                                 unsigned char* h2, unsigned char* h3,
                                                 unsigned char* h4) {
    return {GetAddress(type, compressed, h1), GetAddress(type, compressed, h2),
            GetAddress(type, compressed, h3), GetAddress(type, compressed, h4)};
  }

  std::string Secp256K1::GetPrivAddress(bool compressed, Int& privKey) {
    std::vector<uint8_t> data = {0x80};
    uint8_t tmp[32];
    privKey.Get32Bytes(tmp);
    data.insert(data.end(), tmp, tmp + 32);
    if (compressed) data.push_back(0x01);

    alignas(64) uint8_t inblock[16][32] = {};
    const uint8_t* in[16] = {};
    uint8_t* out[16] = {};
    memcpy(inblock[0], data.data(), data.size());
    in[0] = inblock[0];
    out[0] = inblock[1];
    sha256_avx512_16blocks(in, out);

    in[0] = inblock[1];
    out[0] = inblock[2];
    sha256_avx512_16blocks(in, out);

    data.insert(data.end(), inblock[2], inblock[2] + 4);

    std::string result;
    const char* base58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    std::vector<uint8_t> encoded;
    for (size_t i = 0; i < data.size(); ++i) {
      int carry = data[i];
      for (size_t j = 0; j < encoded.size(); ++j) {
        carry += encoded[j] * 256;
        encoded[j] = carry % 58;
        carry /= 58;
      }
      while (carry > 0) {
        encoded.push_back(carry % 58);
        carry /= 58;
      }
    }

    for (size_t i = 0; i < data.size() && data[i] == 0; ++i) {
      result += base58[0];
    }

    for (int i = encoded.size() - 1; i >= 0; --i) {
      result += base58[encoded[i]];
    }

    return result;
  }

  std::string Secp256K1::GetPublicKeyHex(bool compressed, Point& p) {
    uint8_t buffer[65];
    SerializePublicKey(p, compressed, buffer);
    int len = compressed ? 33 : 65;

    std::stringstream ss;
    for (int i = 0; i < len; i++) {
      ss << std::hex << std::setw(2) << std::setfill('0') << (int)buffer[i];
    }
    return ss.str();
  }

  Point Secp256K1::ParsePublicKeyHex(std::string str, bool& isCompressed) {
    Point p;
    if (str.length() == 66) {
      isCompressed = true;
      uint8_t prefix = GetByte(str, 0);
      bool isEven = (prefix == 0x02);

      Int x;
      std::string xStr = str.substr(2);
      x.SetBase16(const_cast<char*>(xStr.c_str()));

      Int y = GetY(x, isEven);

      p.x = x;
      p.y = y;
      p.z.SetInt32(1);
    } else if (str.length() == 130) {
      isCompressed = false;
      std::string xStr = str.substr(2, 64);
      std::string yStr = str.substr(66, 64);

      p.x.SetBase16(const_cast<char*>(xStr.c_str()));
      p.y.SetBase16(const_cast<char*>(yStr.c_str()));
      p.z.SetInt32(1);
    }

    return p;
  }

  bool Secp256K1::CheckPudAddress(std::string address) { return true; }

  Int Secp256K1::DecodePrivateKey(char* key, bool* compressed) {
    *compressed = false;

    static const char* base58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::vector<uint8_t> num;
    for (char* p = key; *p; ++p) {
      const char* q = strchr(base58, *p);
      if (!q) return Int((uint64_t)0);
      num.push_back(q - base58);
    }
    std::vector<uint8_t> data;
    for (size_t i = 0; i < num.size(); ++i) {
      int carry = num[i];
      for (size_t j = 0; j < data.size(); ++j) {
        carry += data[j] * 58;
        data[j] = carry & 0xFF;
        carry >>= 8;
      }
      while (carry > 0) {
        data.push_back(carry & 0xFF);
        carry >>= 8;
      }
    }
    if (data.size() < 37) return Int((uint64_t)0);

    std::reverse(data.begin(), data.end());
    if (data[0] != 0x80) return Int((uint64_t)0);

    if (data.size() == 38 && data[33] == 0x01) {
      *compressed = true;
    }

    Int result;
    result.Set32Bytes(data.data() + 1);
    return result;
  }

  Point Secp256K1::Add(Point & p1, Point & p2) { return AddDirect(p1, p2); }

  Point Secp256K1::Add2(Point & p1, Point & p2) { return AddDirect(p1, p2); }

  Point Secp256K1::AddDirect(Point & p1, Point & p2) {
    Int _s;
    Int _p;
    Int dy;
    Int dx;
    Point r;
    r.z.SetInt32(1);

    dy.ModSub(&p2.y, &p1.y);
    dx.ModSub(&p2.x, &p1.x);
    dx.ModInv();
    _s.ModMulK1(&dy, &dx);

    _p.ModSquareK1(&_s);

    r.x.ModSub(&_p, &p1.x);
    r.x.ModSub(&p2.x);

    r.y.ModSub(&p2.x, &r.x);
    r.y.ModMulK1(&_s);
    r.y.ModSub(&p2.y);

    return r;
  }

  Point Secp256K1::Double(Point & p) { return DoubleDirect(p); }

  Point Secp256K1::DoubleDirect(Point & p) {
    Int _s;
    Int _p;
    Int a;
    Point r;
    r.z.SetInt32(1);

    _s.ModMulK1(&p.x, &p.x);
    _p.ModAdd(&_s, &_s);
    _p.ModAdd(&_s);

    a.ModAdd(&p.y, &p.y);
    a.ModInv();
    _s.ModMulK1(&_p, &a);

    _p.ModMulK1(&_s, &_s);
    a.ModAdd(&p.x, &p.x);
    a.ModNeg();
    r.x.ModAdd(&a, &_p);

    a.ModSub(&r.x, &p.x);

    _p.ModMulK1(&a, &_s);
    r.y.ModAdd(&_p, &p.y);
    r.y.ModNeg();

    return r;
  }

  Point Secp256K1::DoubleDirect_Safe(Point & p) { return DoubleDirect(p); }

  Point Secp256K1::ComputePublicKey(Int * privKey) {
    int i = 0;
    uint8_t b;
    Point Q;
    Q.Clear();

    for (i = 0; i < 32; i++) {
      b = privKey->GetByte(i);
      if (b) break;
    }
    Q = GTable[256 * i + (b - 1)];
    i++;

    for (; i < 32; i++) {
      b = privKey->GetByte(i);
      if (b) Q = Add2(Q, GTable[256 * i + (b - 1)]);
    }

    Q.Reduce();
    return Q;
  }

  Point Secp256K1::NextKey(Point & key) { return Add(key, G); }

  void Secp256K1::Check() {
    Point p = ComputePublicKey(&order);
    if (!p.isZero()) {
      std::cout << "Warning: Public key computation check failed" << std::endl;
    }
  }

  bool Secp256K1::EC(Point & p) {
    Int _s;
    Int _p;

    _s.ModSquareK1(&p.x);
    _p.ModMulK1(&_s, &p.x);
    _p.ModAdd(7);
    _s.ModMulK1(&p.y, &p.y);
    _s.ModSub(&_p);

    return _s.IsZero();
  }

  Int Secp256K1::GetY(Int x, bool isEven) {
    Int _s;
    Int _p;

    _s.ModSquareK1(&x);
    _p.ModMulK1(&_s, &x);
    _p.ModAdd(7);
    _p.ModSqrt();

    if (!_p.IsEven() && isEven) {
      _p.ModNeg();
    } else if (_p.IsEven() && !isEven) {
      _p.ModNeg();
    }

    return _p;
  }

  uint8_t Secp256K1::GetByte(std::string & str, int idx) {
    char tmp[3];
    tmp[0] = str[2 * idx];
    tmp[1] = str[2 * idx + 1];
    tmp[2] = 0;
    return (uint8_t)strtol(tmp, nullptr, 16);
  }
