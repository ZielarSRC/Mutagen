#ifndef SECP256K1H
#define SECP256K1H

#include "Point.h"
#include <vector>

#define P2PKH  0
#define P2SH   1
#define BECH32 2

class Secp256K1 {
public:
    Secp256K1();
    ~Secp256K1();

    void Init();
    Point ComputePublicKey(Int *privKey);
    Point NextKey(Point &key);
    void Check();
    bool EC(Point &p);
    Int GetY(Int x, bool isEven);

    // AVX-512 accelerated methods
    __attribute__((target("avx512f"))) Point Add(Point &p1, Point &p2);
    __attribute__((target("avx512f"))) Point Add2(Point &p1, Point &p2);
    __attribute__((target("avx512f"))) Point AddDirect(Point &p1, Point &p2);
    __attribute__((target("avx512f"))) Point Double(Point &p);
    __attribute__((target("avx512f"))) Point DoubleDirect(Point &p);

    // Utility methods
    void GetHash160(int type, bool compressed, Point &pubKey, unsigned char *hash);
    std::string GetAddress(int type, bool compressed, Point &pubKey);
    std::string GetPrivAddress(bool compressed, Int &privKey);
    static Int DecodePrivateKey(char *key, bool *compressed);

    // Generator table with 64-byte alignment for AVX-512
    alignas(64) Point GTable[256*32];
    Int order;
    Point G;

private:
    uint8_t GetByte(std::string &str, int idx);
};

#endif // SECP256K1H