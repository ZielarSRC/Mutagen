#include "Point.h"

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
  x.Set(&p.x);
  y.Set(&p.y);
  z.Set(&p.z);
}

Point &Point::operator=(const Point &p) {
  x.Set((Int *)&p.x);
  y.Set((Int *)&p.y);
  z.Set((Int *)&p.z);
  return *this;
}

bool Point::IsZero() { return x.IsZero() && y.IsZero(); }

bool Point::Equal(Point &p) { return x.IsEqual(&p.x) && y.IsEqual(&p.y); }

Point Point::Neg() {
  Point r;
  r.x.Set(&this->x);
  r.y.Neg();
  r.y.Add(&secp256k1->P);
  r.z.SetInt32(1);
  return r;
}

void Point::Normalize() {
  if (z.IsOne()) return;

  Int zi;
  zi.ModInv(&z, &secp256k1->P);

  Int zi2;
  zi2.ModSquare(&zi, &secp256k1->P);

  x.ModMul(&x, &zi2, &secp256k1->P);
  y.ModMul(&y, &zi2, &secp256k1->P);
  y.ModMul(&y, &zi, &secp256k1->P);

  z.SetInt32(1);
}

Point Point::DoubleDirect() {
  Int _3;
  _3.SetInt32(3);

  Int a;
  Int b;
  Int c;
  Int d;
  Int e;
  Int f;

  // XX = X1^2
  a.ModSquare(&x, &secp256k1->P);

  // YY = Y1^2
  b.ModSquare(&y, &secp256k1->P);

  // ZZ = Z1^2
  c.ModSquare(&z, &secp256k1->P);

  // S = 4*X1*YY
  d.ModMul(&x, &b, &secp256k1->P);
  d.ModDouble(&secp256k1->P);
  d.ModDouble(&secp256k1->P);

  // M = 3*XX+a*ZZ^2
  e.ModMul(&_3, &a, &secp256k1->P);

  // T = M^2-2*S
  f.ModSquare(&e, &secp256k1->P);
  f.ModSub(&f, &d, &secp256k1->P);
  f.ModSub(&f, &d, &secp256k1->P);

  Point r;

  // X3 = T
  r.x.Set(&f);

  // Y3 = M*(S-T)-8*YY^2
  b.ModSquare(&b, &secp256k1->P);
  b.ModDouble(&secp256k1->P);
  b.ModDouble(&secp256k1->P);
  b.ModDouble(&secp256k1->P);
  d.ModSub(&d, &f, &secp256k1->P);
  e.ModMul(&e, &d, &secp256k1->P);
  e.ModSub(&e, &b, &secp256k1->P);
  r.y.Set(&e);

  // Z3 = 2*Y1*Z1
  r.z.ModMul(&y, &z, &secp256k1->P);
  r.z.ModDouble(&secp256k1->P);

  return r;
}

void Point::Init(Int *p, Int *s) {
  secp256k1 = new Secp256K1();
  secp256k1->Init();
}

bool Point::EC(Int &n) {
  Int y2;
  Int x3;

  // y^2 = x^3 + 7
  y2.ModSquare(&y, &n);
  x3.ModMul(&x, &x, &n);
  x3.ModMul(&x3, &x, &n);
  x3.ModAdd(&x3, Int(7), &n);

  return y2.IsEqual(&x3);
}
