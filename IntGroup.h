#ifndef INTGROUPH
#define INTGROUPH

#include "Int.h"

class IntGroup {
 public:
  IntGroup(int size);
  ~IntGroup();

  void Set(Int *pts);
  __attribute__((target("avx512f"))) void ModInv();  // AVX-512 batch inversion
  Int Get(Int *a);
  void InitP();

 private:
  Int *ints;
  Int *subp;
  int size;
  bool initialized;
  Int P;
};

#endif  // INTGROUPH
