#include "Point.h"

Point::Point() {}

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
}

void Point::Clear() {
// Zoptymalizowane czyszczenie - wszystkie operacje jednocześnie
#pragma omp parallel sections
  {
#pragma omp section
    { x.SetInt32(0); }
#pragma omp section
    { y.SetInt32(0); }
#pragma omp section
    { z.SetInt32(0); }
  }
}

void Point::Set(Int *cx, Int *cy, Int *cz) {
// Zoptymalizowane ustawianie - wszystkie operacje jednocześnie
#pragma omp parallel sections
  {
#pragma omp section
    { x.Set(cx); }
#pragma omp section
    { y.Set(cy); }
#pragma omp section
    { z.Set(cz); }
  }
}

Point::~Point() {
  // Destruktor nie wymaga zmian
}

void Point::Set(const Point &p) {
// Zoptymalizowane kopiowanie - wszystkie operacje jednocześnie
#pragma omp parallel sections
  {
#pragma omp section
    { x.Set(&p.x); }
#pragma omp section
    { y.Set(&p.y); }
#pragma omp section
    { z.Set(&p.z); }
  }
}

Point &Point::operator=(const Point &p) {
  // Zoptymalizowany operator przypisania
  if (this != &p) {
    Set(p);
  }
  return *this;
}

bool Point::IsZero() const { return x.IsZero() && y.IsZero(); }

bool Point::IsInfinity() const { return z.IsZero(); }

bool Point::Equals(const Point &p) const {
  // Sprawdzenie czy punkty są identyczne
  return x.IsEqual(&p.x) && y.IsEqual(&p.y) && z.IsEqual(&p.z);
}

bool Point::operator==(const Point &p) const { return Equals(p); }

void Point::Reduce() {
  // Standardowa redukcja
  if (z.IsZero()) return;  // Punkt w nieskończoności

  Int i(&z);
  i.ModInv();
  x.ModMul(&x, &i);
  y.ModMul(&y, &i);
  z.SetInt32(1);
}

void Point::ReduceFast() {
  // Zoptymalizowana redukcja dla AVX-512
  if (z.IsZero()) return;  // Punkt w nieskończoności

  Int zInv;
  ComputeZ1Inv(zInv);

// Zrównoleglone operacje ModMul
#pragma omp parallel sections
  {
#pragma omp section
    {
      Int zInv2;
      zInv2.ModSquare(&zInv);
      x.ModMul(&x, &zInv2);
    }

#pragma omp section
    {
      Int zInv3;
      zInv3.ModMul(&zInv, &zInv);
      zInv3.ModMul(&zInv3, &zInv);
      y.ModMul(&y, &zInv3);
    }
  }

  z.SetInt32(1);
}

void Point::ComputeZ1Inv(Int &zInv) const {
  // Optymalizacja inversion trick dla AVX-512
  zInv.Set(&z);
  zInv.ModInv();
}

void Point::Normalize() {
  // Alias dla Reduce() dla lepszej czytelności kodu
  Reduce();
}

void Point::Prefetch(int hint) const {
  // Prefetching danych do pamięci podręcznej
  _mm_prefetch((const char *)&x, hint);
  _mm_prefetch((const char *)&y, hint);
  _mm_prefetch((const char *)&z, hint);
}

bool Point::IsOnCurve() const {
  // Sprawdzenie równania krzywej y^2 = x^3 + 7 (dla secp256k1)
  if (IsInfinity()) return true;

  Int y2, x3, temp;

  // Jeśli z != 1, musimy znormalizować punkt
  if (!z.IsOne()) {
    Point normalized(*this);
    normalized.Normalize();
    return normalized.IsOnCurve();
  }

  y2.ModSquare(&y);
  x3.ModSquare(&x);
  x3.ModMul(&x3, &x);

  temp.SetInt32(7);  // a = 0, b = 7 dla secp256k1
  x3.ModAdd(&temp);

  return y2.IsEqual(&x3);
}

void Point::BatchReduce(Point *points, int count) {
  if (count <= 1) {
    if (count == 1) points[0].Reduce();
    return;
  }

  // Implementacja Montgomery batch inversion trick
  Int *zs = new Int[count];
  Int *temps = new Int[count];

  // Obliczanie z-wartości
  for (int i = 0; i < count; i++) {
    zs[i].Set(&points[i].z);
  }

  // Akumulacja produktów
  temps[0].Set(&zs[0]);
  for (int i = 1; i < count; i++) {
    temps[i].ModMul(&temps[i - 1], &zs[i]);
  }

  // Inwersja akumulatora
  Int acc;
  acc.Set(&temps[count - 1]);
  acc.ModInv();

  // Obliczanie inwersji indywidualnych
  for (int i = count - 1; i > 0; i--) {
    temps[i].ModMul(&acc, &temps[i - 1]);
    acc.ModMul(&acc, &zs[i]);
  }
  temps[0].Set(&acc);

// Zastosowanie inwersji do punktów
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    points[i].x.ModMul(&points[i].x, &temps[i]);
    points[i].y.ModMul(&points[i].y, &temps[i]);
    points[i].z.SetInt32(1);
  }

  delete[] zs;
  delete[] temps;
}

void Point::BatchNormalize(Point *points, int count) {
  // Alias dla BatchReduce
  BatchReduce(points, count);
}

void PointBatchOperation(Point *points, int count, void (*operation)(Point &)) {
// Funkcja do równoległego wykonywania operacji na wielu punktach
#pragma omp parallel for
  for (int i = 0; i < count; i++) {
    operation(points[i]);
  }
}
