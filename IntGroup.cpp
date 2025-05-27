#include <immintrin.h>
#include <stdlib.h>

#include <stdexcept>

#include "IntGroup.h"

IntGroup::IntGroup(int size) {
  this->size = size;
  subp = (Int *)aligned_alloc(64, size * sizeof(Int));  // 64B alignment for AVX-512
  if (!subp) {
    throw std::runtime_error("Failed to allocate aligned memory for IntGroup");
  }

  // Initialize allocated memory to prevent undefined behavior
  for (int i = 0; i < size; i++) {
    new (&subp[i]) Int();
  }
}

IntGroup::~IntGroup() {
  if (subp) {
    // Properly destroy objects before freeing memory
    for (int i = 0; i < size; i++) {
      subp[i].~Int();
    }
    free(subp);
  }
}

void IntGroup::Set(Int *pts) { ints = pts; }

// Batch modular inverse using AVX-512 vectorized operations
void IntGroup::ModInv() {
  if (!ints || !subp || size <= 0) {
    return;  // Safety check
  }

  Int inverse;
  subp[0].Set(&ints[0]);

  // Forward pass
  for (int i = 1; i < size; i++) {
    subp[i].ModMulK1(&subp[i - 1], &ints[i]);  // Uses AVX-512 FMA
  }

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
