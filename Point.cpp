#include <immintrin.h>

#include <cstring>

#include "Point.h"
#include "SECP256K1.h"

Secp256K1 *secp256k1 = NULL;

Point::Point() { Clear(); }

Point::Point(const Point &p) {
  x.Set((Int *)&p.x);
  y.Set((Int *)&p.y);
  z.Set((Int *)&p.z);
}

Point::Point(Int *cx, Int *cy, Int *cz) {
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::Point(Int *cx, Int *cy) {
  x.Set(cx);
  y.Set(cy);
  z.SetInt32(1);
}

void Point::Clear() {
  x.SetInt32(0);
  y.SetInt32(0);
  z.SetInt32(0);
}

void Point::Set(Int *x, Int *y) {
  this->x.Set(x);
  this->y.Set(y);
  this->z.SetInt32(1);
}

void Point::Set(Int *x, Int *y, Int *z) {
  this->x.Set(x);
  this->y.Set(y);
  this->z.Set(z);
}

void Point::Set(Point &p) {
  this->x.Set(&p.x);
  this->y.Set(&p.y);
  this->z.Set(&p.z);
}

Point &Point::operator=(const Point &p) {
  x.Set((Int *)&p.x);
  y.Set((Int *)&p.y);
  z.Set((Int *)&p.z);
  return *this;
}

bool Point::IsZero() const { return x.IsZero() && y.IsZero(); }

bool Point::IsEqual(const Point &p) const {
  if (z.IsZero() && p.z.IsZero()) return true;

  if (z.IsZero() || p.z.IsZero()) return false;

  if (z.IsOne() && p.z.IsOne()) {
    return x.IsEqual(&p.x) && y.IsEqual(&p.y);
  }

  // X1*Z2^2 = X2*Z1^2
  // Y1*Z2^3 = Y2*Z1^3

  Int z1z1;
  Int z2z2;
  z1z1.ModSquare(&z, &secp256k1->P);
  z2z2.ModSquare((Int *)&p.z, &secp256k1->P);

  Int u1;
  Int u2;
  u1.ModMul(&x, &z2z2, &secp256k1->P);
  u2.ModMul((Int *)&p.x, &z1z1, &secp256k1->P);

  if (!u1.IsEqual(&u2)) return false;

  Int z1z1z1;
  Int z2z2z2;
  z1z1z1.ModMul(&z1z1, &z, &secp256k1->P);
  z2z2z2.ModMul(&z2z2, (Int *)&p.z, &secp256k1->P);

  Int s1;
  Int s2;
  s1.ModMul(&y, &z2z2z2, &secp256k1->P);
  s2.ModMul((Int *)&p.y, &z1z1z1, &secp256k1->P);

  return s1.IsEqual(&s2);
}

Point Point::Neg() {
  Point r;
  r.x.Set(&this->x);
  r.y.Set(&this->y);
  r.z.Set(&this->z);
  r.y.ModNeg();
  return r;
}

void Point::Normalize() {
  if (z.IsZero()) {
    Clear();
    return;
  }

  if (z.IsOne()) return;

  Int zi;
  zi.ModInv(&z);

  Int zi2;
  zi2.ModSquare(&zi);

  Int zi3;
  zi3.ModMul(&zi2, &zi);

  x.ModMul(&x, &zi2);
  y.ModMul(&y, &zi3);
  z.SetInt32(1);
}

void Point::Reduce() {
  x.Mod(&secp256k1->P);
  y.Mod(&secp256k1->P);
  z.Mod(&secp256k1->P);
}

void Point::Affine() {
  Normalize();
  Reduce();
}

bool Point::EC() {
  // y^2 = x^3 + 7
  // y^2 - x^3 - 7 = 0

  if (IsZero()) return true;

  Int y2;
  Int x3;
  Int bn;

  y2.ModSquare(&y);
  x3.ModSquare(&x);
  x3.ModMul(&x3, &x);
  x3.ModAdd(&secp256k1->B);

  bn.ModSub(&y2, &x3);

  return bn.IsZero();
}

Point Point::DoubleDirect() {
  // https://hyperelliptic.org/EFD/g1p/auto-shortw-jacobian.html
  // dbl-2007-bl

  // Cost: 1M + 8S + 1*a + 10add + 2*2 + 1*3 + 1*8

  if (IsZero()) {
    return *this;
  }

  Int XX;
  Int YY;
  Int YYYY;
  Int ZZ;
  Int S;
  Int M;
  Int T;

  XX.ModSquare(&x);
  YY.ModSquare(&y);
  YYYY.ModSquare(&YY);
  ZZ.ModSquare(&z);

  // S = 2*((X1+YY)^2-XX-YYYY)
  S.ModAdd(&x, &YY);
  S.ModSquare(&S);
  S.ModSub(&XX);
  S.ModSub(&YYYY);
  S.ModAdd(&S, &S);

  // M = 3*XX+a*ZZ^2
  M.ModAdd(&XX, &XX);
  M.ModAdd(&XX, &M);

  // T = M^2-2*S
  T.ModSquare(&M);
  T.ModSub(&S);
  T.ModSub(&S);

  Point R;

  // X3 = T
  R.x.Set(&T);

  // Y3 = M*(S-T)-8*YYYY
  YYYY.ModDouble();
  YYYY.ModDouble();
  YYYY.ModDouble();
  S.ModSub(&T);
  R.y.ModMul(&M, &S);
  R.y.ModSub(&YYYY);

  // Z3 = (Y1+Z1)^2-YY-ZZ
  R.z.ModAdd(&y, &z);
  R.z.ModSquare(&R.z);
  R.z.ModSub(&YY);
  R.z.ModSub(&ZZ);

  return R;
}

Point Point::Double() {
  // 2P = P+P
  if (IsZero()) {
    return *this;
  }

  if (y.IsZero()) {
    // P = (x,0)
    // 2P = O (infinity)
    Point r;
    r.Clear();
    return r;
  }

  return DoubleDirect();
}

Point Point::Add(const Point &p) {
  // P + Q (general case)
  if (p.IsZero()) return *this;

  if (IsZero()) return p;

  if (IsEqual(p)) {
    return Double();
  }

  return Add2(p);
}

Point Point::Add2(const Point &p) {
  // P1 + P2 (P1!=P2, P1!=-P2)

  // Complete addition formula for a = 0 (Jacobian)
  // http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl

  // Suitability: Suitable for fixed single coordinate.

  Int Z1Z1;
  Int Z2Z2;
  Int U1;
  Int U2;
  Int S1;
  Int S2;
  Int H;
  Int I;
  Int J;
  Int r;
  Int V;

  Z1Z1.ModSquare(&z);
  Z2Z2.ModSquare((Int *)&p.z);

  U1.ModMul(&x, &Z2Z2);
  U2.ModMul((Int *)&p.x, &Z1Z1);

  S1.ModMul(&y, (Int *)&p.z);
  S1.ModMul(&S1, &Z2Z2);

  S2.ModMul((Int *)&p.y, &z);
  S2.ModMul(&S2, &Z1Z1);

  H.ModSub(&U2, &U1);

  // If H=0 then this is either doubling or point at infinity
  // But we ruled out those cases earlier

  I.ModAdd(&H, &H);
  I.ModSquare(&I);

  J.ModMul(&H, &I);

  r.ModSub(&S2, &S1);
  r.ModAdd(&r, &r);

  V.ModMul(&U1, &I);

  Point R;

  // X3 = r^2 - J - 2*V
  R.x.ModSquare(&r);
  R.x.ModSub(&J);
  R.x.ModSub(&V);
  R.x.ModSub(&V);

  // Y3 = r*(V-X3)-2*S1*J
  V.ModSub(&R.x);
  R.y.ModMul(&r, &V);
  S1.ModMul(&S1, &J);
  S1.ModAdd(&S1, &S1);
  R.y.ModSub(&S1);

  // Z3 = ((Z1+Z2)^2-Z1Z1-Z2Z2)*H
  R.z.ModAdd(&z, (Int *)&p.z);
  R.z.ModSquare(&R.z);
  R.z.ModSub(&Z1Z1);
  R.z.ModSub(&Z2Z2);
  R.z.ModMul(&R.z, &H);

  return R;
}
