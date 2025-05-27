#ifndef POINTH
#define POINTH

#include "Int.h"

class Point {
 public:
  Point();
  Point(const Point &p);
  Point(Int *cx, Int *cy, Int *cz);
  Point(Int *cx, Int *cy);

  void Clear();
  void Set(Int *x, Int *y);
  void Set(Int *x, Int *y, Int *z);
  void Set(Point &p);
  bool IsZero();
  bool Equal(Point &p);
  Point &operator=(const Point &p);
  Point Neg();

  // ECDSA
  bool EC(Int &n);

  // Point operations
  Point DoubleDirect();
  void Normalize();
  Point Neg();

  // Initialize with generator parameters
  void Init(Int *p, Int *s);

  Int x;
  Int y;
  Int z;
};

#endif  // POINTH
