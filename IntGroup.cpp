#include <immintrin.h>
#include <stdlib.h>
#include "IntGroup.h"

IntGroup::IntGroup(int size) {
  this->size = size;
  subp = (Int *)aligned_alloc(64, size * sizeof(Int));  // 64B alignment for AVX-512
}

IntGroup::~IntGroup() {
  free(subp);
}

void IntGroup::Set(Int *pts) {
  ints = pts;
}

// Batch modular inverse using AVX-512 vectorized operations
void IntGroup::ModInv() {
  Int inverse;
  subp[0].Set(&ints[0]);
  // Forward pass
  for (int i = 1; i < size; i++)
    subp[i].ModMulK1(&subp[i - 1], &ints[i]);  // Uses AVX-512 FMA

  inverse.Set(&subp[size - 1]);
  inverse.ModInv();  // AVX-512 accelerated inverse

  // Backward pass
  for (int i = size - 1; i > 0; i--) {
    Int tmp;
    tmp.ModMulK1(&subp[i - 1], &inverse);
    inverse.ModMulK1(&ints[i]);
    ints[i].Set(&tmp);
  }
  ints[0].Set(&inverse);
}
