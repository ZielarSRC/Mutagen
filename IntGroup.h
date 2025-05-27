#ifndef INTGROUPH
#define INTGROUPH

#include <immintrin.h>

#include <vector>

#include "Int.h"

class IntGroup {
 public:
  IntGroup(int size);
  ~IntGroup();
  void Set(Int* a);
  void ModInv();
  Int Get(Int* a);
  void InitP();

 private:
  int size;
  bool initialized;
  Int* table;
  Int P;
};

#endif  // INTGROUPH
