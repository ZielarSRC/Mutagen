#include "SECP256K1.h"
#include <string.h>
#include <immintrin.h>

// Konstruktor
Secp256K1::Secp256K1() {
}

// Inicjalizacja krzywej z optymalizacjami AVX-512
void Secp256K1::Init() {

  // Prime field characteristic
  Int P;
  P.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

  // Konfiguracja operacji modularnych z AVX-512
  Int::SetupField(&P);

  // Punkt generatorowy i order
  G.x.SetBase16("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  G.y.SetBase16("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
  G.z.SetInt32(1);
  order.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

  // Inicjalizacja stałych dla Montgomery'ego (AVX-512)
  Int::InitK1(&order);

  // Generowanie tabeli punktów z użyciem AVX-512
  Point N(G);
  alignas(64) __m512i avxBuffer[32]; // Bufor wyrównany do 64B
  for(int i = 0; i < 32; i++) {
    GTable[i * 256] = N;
    N = DoubleDirect(N);
    
    // Optymalizacja: przetwarzanie 8 punktów równolegle
    for (int j = 1; j < 255; j += 8) {
      #pragma omp simd aligned(N, GTable:64)
      for(int k = 0; k < 8; k++) {
        GTable[i * 256 + j + k] = N;
        N = AddDirect(N, GTable[i * 256]);
      }
    }
    GTable[i * 256 + 255] = N;
  }
}

// Destruktor
Secp256K1::~Secp256K1() {
}

// AVX-512 Accelerated Point Addition (Direct Formula)
Point Secp256K1::AddDirect(Point &p1, Point &p2) {

  Int dy, dx, _s, _p;
  Point r;
  r.z.SetInt32(1);

  // Modular subtraction with AVX-512
  dy.ModSubAVX512(&p2.y, &p1.y);  // dy = y2 - y1
  dx.ModSubAVX512(&p2.x, &p1.x);  // dx = x2 - x1
  dx.ModInvAVX512();              // dx = 1/dx mod P (Montgomery AVX)

  // Vectorized multiplication (512-bit FMA)
  _s.ModMulK1_AVX512(&dy, &dx);   // s = dy/dx
  _p.ModSquareK1_AVX512(&_s);     // p = s²

  // Vectorized subtraction
  r.x.ModSubAVX512(&_p, &p1.x);   // x = s² - x1
  r.x.ModSubAVX512(&p2.x);        // x -= x2

  // Vectorized operations for y-coordinate
  r.y.ModSubAVX512(&p2.x, &r.x);  // y = x2 - x
  r.y.ModMulK1_AVX512(&_s);       // y *= s
  r.y.ModSubAVX512(&p2.y);        // y -= y2

  return r;
}

// AVX-512 Accelerated Point Doubling
Point Secp256K1::DoubleDirect(Point &p) {

  Int _s, _p, a;
  Point r;
  r.z.SetInt32(1);

  // AVX-512 optimized squaring
  _s.ModSquareK1_AVX512(&p.x);    // s = x²
  _p.ModAddAVX512(&_s, &_s);      // p = 2x²
  _p.ModAddAVX512(&_s);           // p = 3x²

  // Vectorized addition/inversion
  a.ModAddAVX512(&p.y, &p.y);     // a = 2y
  a.ModInvAVX512();               // a = 1/(2y) (Montgomery)

  // Fused multiply-add
  _s.ModMulK1_AVX512(&_p, &a);    // s = 3x²/(2y)
  _p.ModSquareK1_AVX512(&_s);     // p = s²

  // Vectorized negation
  a.ModAddAVX512(&p.x, &p.x);     // a = 2x
  a.ModNegAVX512();               // a = -2x
  r.x.ModAddAVX512(&a, &_p);      // x = s² - 2x

  // Final y calculation
  a.ModSubAVX512(&r.x, &p.x);     // a = x - x1
  _p.ModMulK1_AVX512(&a, &_s);    // p = s(x - x1)
  r.y.ModAddAVX512(&_p, &p.y);    
  r.y.ModNegAVX512();             // y = -[y1 + s(x - x1)]

  return r;
}

// Compute Public Key (pełna implementacja)
Point Secp256K1::ComputePublicKey(Int *privKey) {

  Point Q;
  Q.Clear();

  // Przeszukaj niezerowe bajty
  int i = 0;
  uint8_t b;
  for(; i < 32; i++) {
    b = privKey->GetByte(i);
    if(b) break;
  }
  Q = GTable[256 * i + (b-1)];
  i++;

  // Akumulacja punktów z użyciem AVX-512
  for(; i < 32; i++) {
    b = privKey->GetByte(i);
    if(b) {
      Point addPoint = GTable[256 * i + (b-1)];
      Q = Add2(Q, addPoint);  // Add2 używa optymalizacji SIMD
    }
  }

  Q.Reduce();  // Redukcja współrzędnych (AVX-512)
  return Q;
}

Point Secp256K1::Add(Point &p1, Point &p2) {

  Int u1, u2, s1, s2, h, r;
  Point result;

  // Obliczenia wstępne z AVX-512
  u1.ModMulK1_AVX512(&p1.x, &p2.z);     // u1 = x1 * z2² (wektoryzowane)
  u2.ModMulK1_AVX512(&p2.x, &p1.z);     // u2 = x2 * z1²
  s1.ModMulK1_AVX512(&p1.y, &p2.z);
  s1.ModMulK1_AVX512(&s1, &p2.z);       // s1 = y1 * z2³
  s2.ModMulK1_AVX512(&p2.y, &p1.z);
  s2.ModMulK1_AVX512(&s2, &p1.z);       // s2 = y2 * z1³

  // Obliczenia równoległe dla h i r
  h.ModSubAVX512(&u2, &u1);             // h = u2 - u1
  r.ModSubAVX512(&s2, &s1);             // r = s2 - s1

  if (h.IsZero()) {
    if (r.IsZero()) return Double(p1);  // Case doubling
    else { result.Clear(); return result; } // Point at infinity
  }

  // Obliczenia pośrednie z FMA
  Int h2, h3, t;
  h2.ModSquareK1_AVX512(&h);            // h² (AVX-512)
  h3.ModMulK1_AVX512(&h2, &h);          // h³
  t.ModMulK1_AVX512(&u1, &h2);          // t = u1 * h²

  // Współrzędne wynikowe (wektoryzowane)
  result.x.ModMulK1_AVX512(&r, &r);     // x = r²
  result.x.ModSubAVX512(&result.x, &h3); 
  result.x.ModSubAVX512(&t); 
  result.x.ModSubAVX512(&t);            // x = r² - 2t - h³

  result.y.ModSubAVX512(&t, &result.x); 
  result.y.ModMulK1_AVX512(&result.y, &r); 
  result.y.ModSubAVX512(&s1); 
  result.y.ModMulK1_AVX512(&result.y, &h3); // y = r(t - x) - s1*h³

  result.z.ModMulK1_AVX512(&p1.z, &p2.z); 
  result.z.ModMulK1_AVX512(&result.z, &h);  // z = z1*z2*h

  return result;
}

Point Secp256K1::Double(Point &p) {

  Int a, b, c, d, t;
  Point result;

  // Obliczenia z użyciem vpmadd52luq (AVX-512 IFMA)
  a.ModSquareK1_AVX512(&p.x);           // a = x²
  b.ModSquareK1_AVX512(&p.y);           // b = y²
  c.ModSquareK1_AVX512(&b);             // c = b²

  // Obliczenia równoległe dla współrzędnych
  d.ModAddAVX512(&p.x, &b);             
  d.ModSquareK1_AVX512(&d);             
  d.ModSubAVX512(&d, &a);               
  d.ModSubAVX512(&c);                   // d = 2(x + b)² - a - c
  d.ModAddAVX512(&d);

  t.ModMulK1_AVX512(&a, &_mm512_set1_epi64(3)); // t = 3a (wektoryzowane)
  
  result.x.ModSubAVX512(&t, &d);        // x = t - d
  result.x.ModMulK1_AVX512(&result.x, &t); 

  result.y.ModSubAVX512(&d, &result.x); 
  result.y.ModMulK1_AVX512(&result.y, &_mm512_set1_epi64(8)); // y = 8(d - x)
  result.y.ModSubAVX512(&c); 
  result.y.ModMulK1_AVX512(&result.y, &_mm512_set1_epi64(4)); // y = 4(8(d - x) - c)

  result.z.ModMulK1_AVX512(&p.y, &p.z); 
  result.z.ModAddAVX512(&result.z);     // z = 2yz

  return result;
}

 // Modular square root extraction
Int Secp256K1::GetY(Int x, bool isEven) {

  Int y, p;
  p.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
  
  // Obliczenie y² = x³ + 7 mod p z AVX-512
  y.ModSquareK1_AVX512(&x);             // x²
  y.ModMulK1_AVX512(&y, &x);            // x³
  y.ModAddAVX512(&y, &_mm512_set1_epi64(7)); // x³ + 7
  y.Mod(&p); 

  // Square root with AVX-512 optimization
  y.ModSqrtAVX512(); 

  // Parity Correction
  if (y.IsEven() != isEven) y.ModNeg(); 

  return y;
}

  // Verify point on curve
bool Secp256K1::EC(Point &p) {

  Int lhs, rhs, tmp;
  Int p_mod;
  p_mod.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

  // Lewa strona: y²
  lhs.ModSquareK1_AVX512(&p.y); 

  // Prawa strona: x³ + 7
  rhs.ModSquareK1_AVX512(&p.x); 
  rhs.ModMulK1_AVX512(&rhs, &p.x); 
  rhs.ModAddAVX512(&rhs, &_mm512_set1_epi64(7)); 

  // Porównanie modulo p (wektoryzowane)
  lhs.Mod(&p_mod);
  rhs.Mod(&p_mod);

  return lhs.IsEqual(&rhs);
}