#ifndef POINTH
#define POINTH

#include "Int.h"

// Forward declaration
class Secp256K1;
extern Secp256K1 *secp256k1;

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
  bool IsZero() const;
  bool IsEqual(const Point &p) const;
  Point &operator=(const Point &p);
  Point Neg();

  void Reduce();
  void Affine();
  bool EC();

  // Optimized methods for Xeon Platinum 8488C
  Point DoubleDirect();
  void Normalize();
  Point Double();
  Point Add(const Point &p);
  Point Add2(const Point &p);

  Int x;
  Int y;
  Int z;
};

#endif  // POINTH
