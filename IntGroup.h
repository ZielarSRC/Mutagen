#ifndef INTGROUPH
#define INTGROUPH

#include "Int.h"

class IntGroup {
 public:
  IntGroup(int size);
  ~IntGroup();

  void Set(Int *pts);
  __attribute__((target("avx512f"))) void ModInv();  // AVX-512 batch inversion

 private:
  Int *ints;
  Int *subp;
  int size;
};

#endif  // INTGROUPH