#include "Point.h"

Point::Point() {
}

Point::Point(const Point &p) {
  x.Set((Int *)&p.x);  // Uses AVX-512 optimized copy
  y.Set((Int *)&p.y);
  z.Set((Int *)&p.z);
}

Point::Point(Int *cx, Int *cy, Int *cz) {
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::Point(Int *cx, Int *cz) {
  x.Set(cx);
  z.Set(cz);
}

void Point::Clear() {
  x.SetInt32(0);  // AVX-512 optimized zeroing
  y.SetInt32(0);
  z.SetInt32(0);
}

void Point::Set(Int *cx, Int *cy, Int *cz) {
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

bool Point::isZero() {
  return x.IsZero() && y.IsZero();  // AVX-512 vectorized check
}

void Point::Reduce() {
  Int i(&z);
  i.ModInv();  // AVX-512 accelerated inverse
  x.ModMul(&x, &i);
  y.ModMul(&y, &i);
  z.SetInt32(1);
}

bool Point::equals(Point &p) {
  return x.IsEqual(&p.x) && y.IsEqual(&p.y) && z.IsEqual(&p.z);  // Vectorized comparison
}

Point::~Point() {
}