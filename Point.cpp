#include "Point.h"

Point::Point() {
  x.SetInt32(0);
  y.SetInt32(0);
  z.SetInt32(0);
}

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

Point::Point(Int *cx, Int *cz) {
  x.Set(cx);
  z.Set(cz);
  y.SetInt32(0);
}

Point::~Point() {}

Point &Point::operator=(const Point &other) {
  if (this != &other) {
    x.Set((Int *)&other.x);
    y.Set((Int *)&other.y);
    z.Set((Int *)&other.z);
  }
  return *this;
}

void Point::Clear() {
  x.SetInt32(0);
  y.SetInt32(0);
  z.SetInt32(0);
}

void Point::Set(Point &p) {
  x.Set(&p.x);
  y.Set(&p.y);
  z.Set(&p.z);
}

void Point::Set(Int *cx, Int *cy, Int *cz) {
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

bool Point::IsZero() { return z.IsZero(); }

bool Point::IsEqual(Point &p) { return x.IsEqual(&p.x) && y.IsEqual(&p.y) && z.IsEqual(&p.z); }

void Point::Reduce() {
  if (z.IsZero()) {
    return;  // Point at infinity
  }

  Int i(&z);
  i.ModInv();
  x.ModMulK1(&x, &i);
  y.ModMulK1(&y, &i);
  z.SetInt32(1);
}
