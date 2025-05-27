#ifndef POINT_H
#define POINT_H

#include "Int.h"

class Point {
 public:
  Int x, y, z;

  Point();
  Point(const Point &p);
  Point(Int *cx, Int *cy, Int *cz);
  Point(Int *cx, Int *cz);
  ~Point();

  bool IsZero();
  bool IsEqual(Point &p);
  void Set(Point &p);
  void Set(Int *cx, Int *cy, Int *cz);
  void Clear();
  void Reduce();

  // Assignment operator
  Point &operator=(const Point &other);

  // AVX-512 optimizations
  friend class Secp256K1;
};

#endif  // POINT_H
