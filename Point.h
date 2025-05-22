#ifndef POINTH
#define POINTH

#include "Int.h"

class Point {
public:
    Int x, y, z;

    Point();
    Point(const Point &p);
    Point(Int *cx, Int *cy, Int *cz);
    Point(Int *cx, Int *cz);
    ~Point();

    bool isZero();
    bool equals(Point &p);
    void Set(Point &p);
    void Set(Int *cx, Int *cy, Int *cz);
    void Clear();
    void Reduce();

    // AVX-512 optimizations
    friend class Secp256K1;
};

#endif // POINTH