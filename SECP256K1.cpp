#include "SECP256K1.h"
#include "ripemd160_avx512.h"
#include "sha256_avx512.h"
#include <cstring>
#include <sstream>
#include <vector>
#include <iomanip>
#include <algorithm>

static Int SECP256K1_P([]{
    Int p;
    p.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    return p;
}());

void Secp256K1::SerializePublicKey(Point &pubKey, bool compressed, uint8_t *out33) {
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
    Int P;
    P.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
    Int::SetupField(&P);
    G.x.SetBase16("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    G.y.SetBase16("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
    G.z.SetInt32(1);
    order.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    Int::InitK1(&order);

    Point N(G);
    for(int i = 0; i < 32; i++) {
        GTable[i * 256] = N;
        N = DoubleDirect(N);
        for (int j = 1; j < 255; j++) {
            GTable[i * 256 + j] = N;
            N = AddDirect(N, GTable[i * 256]);
        }
        GTable[i * 256 + 255] = N;
    }
}

// --------- AVX-512 batch hash160 helpers ---------

void Secp256K1::GetHash160_Batch16(int type, bool compressed,
    Point* pubkeys[16], uint8_t* hashes[16])
{
    alignas(64) uint8_t pubkey_ser[16][33];
    const uint8_t* in[16];
    uint8_t* out[16];
    for(int i=0; i<16; ++i) {
        SerializePublicKey(*pubkeys[i], compressed, pubkey_ser[i]);
        in[i] = pubkey_ser[i];
        out[i] = hashes[i];
    }
    alignas(64) uint8_t sha[16][32];
    const uint8_t* sha_in[16];
    uint8_t* sha_out[16];
    for(int i=0; i<16; ++i) { sha_in[i]=in[i]; sha_out[i]=sha[i]; }
    sha256_avx512_16blocks(sha_in, sha_out);
    ripemd160_avx512_16blocks((const uint8_t**)sha, out);
}

void Secp256K1::GetHash160_Batch8(int type, bool compressed,
    Point* pubkeys[8], uint8_t* hashes[8])
{
    alignas(64) uint8_t pubkey_ser[16][33] = {};
    const uint8_t* in[16] = {};
    uint8_t* out[16] = {};
    for(int i=0; i<8; ++i) {
        SerializePublicKey(*pubkeys[i], compressed, pubkey_ser[i]);
        in[i] = pubkey_ser[i];
        out[i] = hashes[i];
    }
    for(int i=8; i<16; ++i) { in[i]=pubkey_ser[i]; out[i]=pubkey_ser[i]; }
    alignas(64) uint8_t sha[16][32];
    const uint8_t* sha_in[16];
    uint8_t* sha_out[16];
    for(int i=0; i<16; ++i) { sha_in[i]=in[i]; sha_out[i]=sha[i]; }
    sha256_avx512_16blocks(sha_in, sha_out);
    ripemd160_avx512_16blocks((const uint8_t**)sha, out);
}

void Secp256K1::GetHash160_Batch4(int type, bool compressed,
    Point* pubkeys[4], uint8_t* hashes[4])
{
    alignas(64) uint8_t pubkey_ser[16][33] = {};
    const uint8_t* in[16] = {};
    uint8_t* out[16] = {};
    for(int i=0; i<4; ++i) {
        SerializePublicKey(*pubkeys[i], compressed, pubkey_ser[i]);
        in[i] = pubkey_ser[i];
        out[i] = hashes[i];
    }
    for(int i=4; i<16; ++i) { in[i]=pubkey_ser[i]; out[i]=pubkey_ser[i]; }
    alignas(64) uint8_t sha[16][32];
    const uint8_t* sha_in[16];
    uint8_t* sha_out[16];
    for(int i=0; i<16; ++i) { sha_in[i]=in[i]; sha_out[i]=sha[i]; }
    sha256_avx512_16blocks(sha_in, sha_out);
    ripemd160_avx512_16blocks((const uint8_t**)sha, out);
}

void Secp256K1::GetHash160(int type, bool compressed, Point &pubKey, unsigned char *hash) {
    alignas(64) uint8_t pubkey_ser[16][33] = {};
    const uint8_t* in[16] = {};
    uint8_t* out[16] = {};
    SerializePublicKey(pubKey, compressed, pubkey_ser[0]);
    in[0] = pubkey_ser[0];
    out[0] = hash;
    for(int i=1; i<16; ++i) { in[i]=pubkey_ser[i]; out[i]=pubkey_ser[i]; }
    alignas(64) uint8_t sha[16][32];
    const uint8_t* sha_in[16];
    uint8_t* sha_out[16];
    for(int i=0; i<16; ++i) { sha_in[i]=in[i]; sha_out[i]=sha[i]; }
    sha256_avx512_16blocks(sha_in, sha_out);
    ripemd160_avx512_16blocks((const uint8_t**)sha, out);
}

std::string Secp256K1::GetAddress(int type, bool compressed, Point &pubKey) {
    uint8_t hash160[20];
    GetHash160(type, compressed, pubKey, hash160);
    return GetAddress(type, compressed, hash160);
}

std::string Secp256K1::GetAddress(int type, bool compressed, unsigned char *hash160) {
    std::vector<uint8_t> addrPrefix;
    if(type == P2PKH) addrPrefix = {0x00};
    else if(type == P2SH) addrPrefix = {0x05};
    else if(type == BECH32) return "";

    std::vector<uint8_t> full(addrPrefix);
    full.insert(full.end(), hash160, hash160 + 20);

    // double sha256 (batch 16x, slot 0)
    alignas(64) uint8_t inblock[16][32] = {};
    const uint8_t* in[16] = {};
    uint8_t* out[16] = {};
    memcpy(inblock[0], full.data(), full.size());
    in[0] = inblock[0];
    out[0] = inblock[1]; // sha1 result
    sha256_avx512_16blocks(in, out);

    in[0] = inblock[1];
    out[0] = inblock[2]; // sha2 result
    sha256_avx512_16blocks(in, out);

    full.insert(full.end(), inblock[2], inblock[2]+4);

    static const char* base58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::string result;
    std::vector<uint8_t> num(full);
    while(!num.empty() && num[0] == 0) {
        result += '1'; num.erase(num.begin());
    }
    while(!num.empty()) {
        int rem = 0;
        std::vector<uint8_t> div;
        for(size_t i=0; i<num.size(); ++i) {
            int t = (rem << 8) + num[i];
            div.push_back(t / 58);
            rem = t % 58;
        }
        result += base58[rem];
        while(!div.empty() && div[0] == 0) div.erase(div.begin());
        num = div;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::vector<std::string> Secp256K1::GetAddress(int type, bool compressed, unsigned char *h1, unsigned char *h2, unsigned char *h3, unsigned char *h4) {
    return {
        GetAddress(type, compressed, h1),
        GetAddress(type, compressed, h2),
        GetAddress(type, compressed, h3),
        GetAddress(type, compressed, h4)
    };
}

std::string Secp256K1::GetPrivAddress(bool compressed, Int &privKey) {
    std::vector<uint8_t> data = {0x80};
    uint8_t tmp[32];
    privKey.Get32Bytes(tmp);
    data.insert(data.end(), tmp, tmp+32);
    if (compressed) data.push_back(0x01);

    // double sha256 (batch 16x, slot 0)
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

    data.insert(data.end(), inblock[2], inblock[2]+4);

    static const char* base58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::string result;
    std::vector<uint8_t> num(data);
    while(!num.empty() && num[0] == 0) {
        result += '1'; num.erase(num.begin());
    }
    while(!num.empty()) {
        int rem = 0;
        std::vector<uint8_t> div;
        for(size_t i=0; i<num.size(); ++i) {
            int t = (rem << 8) + num[i];
            div.push_back(t / 58);
            rem = t % 58;
        }
        result += base58[rem];
        while(!div.empty() && div[0] == 0) div.erase(div.begin());
        num = div;
    }
    std::reverse(result.begin(), result.end());
    return result;
}

std::string Secp256K1::GetPublicKeyHex(bool compressed, Point &p) {
    uint8_t buff[65];
    SerializePublicKey(p, compressed, buff);
    std::ostringstream oss;
    size_t len = compressed ? 33 : 65;
    for(size_t i=0; i<len; ++i)
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)buff[i];
    return oss.str();
}

Point Secp256K1::ParsePublicKeyHex(std::string str, bool &isCompressed) {
    Point p;
    if(str.size() == 66 && (str[0] == '0' && (str[1] == '2' || str[1] == '3'))) {
        isCompressed = true;
        std::string hex = str.substr(2);
        uint8_t buff[32];
        for(int i=0; i<32; ++i)
            buff[i] = std::stoi(hex.substr(i*2,2), nullptr, 16);
        p.x.Set32Bytes(buff);
        p.y = GetY(p.x, str[1] == '2');
        p.z.SetInt32(1);
    } else if(str.size() == 130 && str.substr(0,2) == "04") {
        isCompressed = false;
        std::string hex = str.substr(2);
        uint8_t buff[64];
        for(int i=0; i<64; ++i)
            buff[i] = std::stoi(hex.substr(i*2,2), nullptr, 16);
        p.x.Set32Bytes(buff);
        p.y.Set32Bytes(buff+32);
        p.z.SetInt32(1);
    }
    return p;
}

bool Secp256K1::CheckPudAddress(std::string address) {
    static const char* base58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::vector<uint8_t> num;
    for(char c : address) {
        const char* p = strchr(base58, c);
        if(!p) return false;
        num.push_back(p - base58);
    }
    std::vector<uint8_t> data;
    for(size_t i=0; i<num.size(); ++i) {
        int carry = num[i];
        for(size_t j=0; j<data.size(); ++j) {
            carry += data[j]*58;
            data[j] = carry & 0xFF;
            carry >>= 8;
        }
        while(carry > 0) {
            data.push_back(carry & 0xFF);
            carry >>= 8;
        }
    }
    while(address.size() > 0 && address[0] == '1') {
        data.push_back(0x00);
        address = address.substr(1);
    }
    if(data.size() < 5) return false;
    std::reverse(data.begin(), data.end());

    // double sha256 (batch 16x, slot 0)
    alignas(64) uint8_t inblock[16][32] = {};
    const uint8_t* in[16] = {};
    uint8_t* out[16] = {};
    memcpy(inblock[0], data.data(), data.size()-4);
    in[0] = inblock[0];
    out[0] = inblock[1];
    sha256_avx512_16blocks(in, out);

    in[0] = inblock[1];
    out[0] = inblock[2];
    sha256_avx512_16blocks(in, out);

    return memcmp(&data[data.size()-4], inblock[2], 4) == 0;
}

Int Secp256K1::DecodePrivateKey(char *key, bool *compressed) {
    static const char* base58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::vector<uint8_t> num;
    for(char *p=key; *p; ++p) {
        const char* q = strchr(base58, *p);
        if(!q) return Int((uint64_t)0);
        num.push_back(q - base58);
    }
    std::vector<uint8_t> data;
    for(size_t i=0; i<num.size(); ++i) {
        int carry = num[i];
        for(size_t j=0; j<data.size(); ++j) {
            carry += data[j]*58;
            data[j] = carry & 0xFF;
            carry >>= 8;
        }
        while(carry > 0) {
            data.push_back(carry & 0xFF);
            carry >>= 8;
        }
    }
    if(data.size() < 37) return Int((uint64_t)0);
    std::reverse(data.begin(), data.end());
    if(compressed) *compressed = (data.size() > 37 && data[33] == 0x01);
    Int priv;
    priv.Set32Bytes(&data[1]);
    return priv;
}

uint8_t Secp256K1::GetByte(std::string &str, int idx) {
    if (idx < 0 || idx >= (int)str.size()) return 0;
    return (uint8_t)str[idx];
}

// --------- ECC ARYTMETYKA (bez zmian; pełne funkcje Add, Double, itp.) ---------

Point Secp256K1::AddDirect(Point &p1, Point &p2) {
    Int _s, _p, dy, dx;
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

Point Secp256K1::DoubleDirect(Point &p) {
    Int _s, _p, a;
    Point r;
    r.z.SetInt32(1);

    _s.ModMulK1(&p.x, &p.x);
    _p.ModAdd(&_s, &_s);
    _p.ModAdd(&_s);

    a.ModAdd(&p.y, &p.y);
    a.ModInv();
    _s.ModMulK1(&_p, &a);
    _p.ModSquareK1(&_s);

    a.ModAdd(&p.x, &p.x);
    a.ModNeg();
    r.x.ModAdd(&a, &_p);

    a.ModSub(&r.x, &p.x);
    _p.ModMulK1(&a, &_s);
    r.y.ModAdd(&_p, &p.y);
    r.y.ModNeg();

    return r;
}

Point Secp256K1::Add2(Point &p1, Point &p2) {
    Int u, v, u1, v1, vs2, vs3, us2, a, us2w, vs2v2, vs3u2, _2vs2v2;
    Point r;

    u1.ModMulK1(&p2.y, &p1.z);
    v1.ModMulK1(&p2.x, &p1.z);
    u.ModSub(&u1, &p1.y);
    v.ModSub(&v1, &p1.x);
    us2.ModSquareK1(&u);
    vs2.ModSquareK1(&v);
    vs3.ModMulK1(&vs2, &v);
    us2w.ModMulK1(&us2, &p1.z);
    vs2v2.ModMulK1(&vs2, &p1.x);
    _2vs2v2.ModAdd(&vs2v2, &vs2v2);
    a.ModSub(&us2w, &vs3);
    a.ModSub(&_2vs2v2);

    r.x.ModMulK1(&v, &a);

    vs3u2.ModMulK1(&vs3, &p1.y);
    r.y.ModSub(&vs2v2, &a);
    r.y.ModMulK1(&r.y, &u);
    r.y.ModSub(&vs3u2);

    r.z.ModMulK1(&vs3, &p1.z);

    return r;
}

Point Secp256K1::Add(Point &p1, Point &p2) {
    Int u1, u2, v1, v2, u, v, w, us2, vs2, vs3, us2w, vs2v2, vs3u2, a, _2vs2v2;
    Point r;

    u1.ModMulK1(&p2.y, &p1.z);
    u2.ModMulK1(&p1.y, &p2.z);
    v1.ModMulK1(&p2.x, &p1.z);
    v2.ModMulK1(&p1.x, &p2.z);

    if (v1.IsEqual(&v2)) {
        if (!u1.IsEqual(&u2)) {
            r.x.SetInt32(0);
            r.y.SetInt32(0);
            r.z.SetInt32(0);
            return r;
        } else {
            return Double(p1);
        }
    }

    u.ModSub(&u1, &u2);
    v.ModSub(&v1, &v2);
    w.ModMulK1(&p1.z, &p2.z);
    us2.ModSquareK1(&u);
    vs2.ModSquareK1(&v);
    vs3.ModMulK1(&vs2, &v);
    us2w.ModMulK1(&us2, &w);
    vs2v2.ModMulK1(&vs2, &v2);
    _2vs2v2.ModAdd(&vs2v2, &vs2v2);
    a.ModSub(&us2w, &vs3);
    a.ModSub(&_2vs2v2);
    r.x.ModMulK1(&v, &a);
    vs3u2.ModMulK1(&vs3, &u2);
    r.y.ModSub(&vs2v2, &a);
    r.y.ModMulK1(&r.y, &u);
    r.y.ModSub(&vs3u2);
    r.z.ModMulK1(&vs3, &w);
    return r;
}

Point Secp256K1::Double(Point &p) {
    Int z2, x2, _3x2, w, s, s2, b, _8b, _8y2s2, y2, h;
    Point r;

    z2.ModSquareK1(&p.z);
    z2.SetInt32(0);
    x2.ModSquareK1(&p.x);
    _3x2.ModAdd(&x2, &x2);
    _3x2.ModAdd(&x2);
    w.ModAdd(&z2, &_3x2);
    s.ModMulK1(&p.y, &p.z);
    b.ModMulK1(&p.y, &s);
    b.ModMulK1(&p.x);
    h.ModSquareK1(&w);
    _8b.ModAdd(&b, &b);
    _8b.ModDouble();
    _8b.ModDouble();
    h.ModSub(&_8b);

    r.x.ModMulK1(&h, &s);
    r.x.ModAdd(&r.x);

    s2.ModSquareK1(&s);
    y2.ModSquareK1(&p.y);
    _8y2s2.ModMulK1(&y2, &s2);
    _8y2s2.ModDouble();
    _8y2s2.ModDouble();
    _8y2s2.ModDouble();

    r.y.ModAdd(&b, &b);
    r.y.ModAdd(&r.y, &r.y);
    r.y.ModSub(&h);
    r.y.ModMulK1(&w);
    r.y.ModSub(&_8y2s2);

    r.z.ModMulK1(&s2, &s);
    r.z.ModDouble();
    r.z.ModDouble();
    r.z.ModDouble();

    return r;
}

Point Secp256K1::ComputePublicKey(Int *privKey) {
    int i = 0;
    uint8_t b;
    Point Q;
    Q.Clear();

    for (i = 0; i < 32; i++) {
        b = privKey->GetByte(i);
        if(b)
            break;
    }
    Q = GTable[256 * i + (b-1)];
    i++;

    for(; i < 32; i++) {
        b = privKey->GetByte(i);
        if(b)
            Q = Add2(Q, GTable[256 * i + (b-1)]);
    }

    Q.Reduce();
    return Q;
}

Point Secp256K1::NextKey(Point &key) {
    return AddDirect(key, G);
}

void Secp256K1::Check() {}

bool Secp256K1::EC(Point &p) {
    Int _s, _p;
    _s.ModSquareK1(&p.x);
    _p.ModMulK1(&_s, &p.x);
    _p.ModAdd(7);
    _s.ModMulK1(&p.y, &p.y);
    _s.ModSub(&_p);
    return _s.IsZero();
}

Int Secp256K1::GetY(Int x, bool isEven) {
    Int _s, _p;
    _s.ModSquareK1(&x);
    _p.ModMulK1(&_s, &x);
    _p.ModAdd(7);
    _p.ModSqrt();
    if(!_p.IsEven() && isEven) {
        _p.ModNeg();
    }
    else if(_p.IsEven() && !isEven) {
        _p.ModNeg();
    }
    return _p;
}
