#ifndef SECP256K1H
#define SECP256K1H

#include "Point.h"
#include <string>
#include <vector>
#include <cstdint>

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

    // AVX-512 batch-accelerated hash160 (16x)
    void GetHash160_Batch16(int type, bool compressed,
        Point* pubkeys[16], uint8_t* hashes[16]);
    void GetHash160_Batch8(int type, bool compressed,
        Point* pubkeys[8], uint8_t* hashes[8]);
    void GetHash160_Batch4(int type, bool compressed,
        Point* pubkeys[4], uint8_t* hashes[4]);
    void GetHash160(int type, bool compressed, Point &pubKey, unsigned char *hash);

    std::string GetAddress(int type, bool compressed, Point &pubKey);
    std::string GetAddress(int type, bool compressed, unsigned char *hash160);
    std::vector<std::string> GetAddress(int type, bool compressed, unsigned char *h1, unsigned char *h2, unsigned char *h3, unsigned char *h4);
    std::string GetPrivAddress(bool compressed, Int &privKey);
    std::string GetPublicKeyHex(bool compressed, Point &p);
    Point ParsePublicKeyHex(std::string str, bool &isCompressed);

    bool CheckPudAddress(std::string address);

    static Int DecodePrivateKey(char *key, bool *compressed);

    Point Add(Point &p1, Point &p2);
    Point Add2(Point &p1, Point &p2);
    Point AddDirect(Point &p1, Point &p2);
    Point Double(Point &p);
    Point DoubleDirect(Point &p);
    Point DoubleDirect_Safe(Point &p);  // DODANE dla Xeona

    Point G;                 // Generator
    Int   order;             // Curve order

    alignas(64) Point GTable[256*32];

private:
    void SerializePublicKey(Point &pubKey, bool compressed, uint8_t *out33);
    uint8_t GetByte(std::string &str,int idx);

};

#endif // SECP256K1H
