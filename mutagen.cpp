#include <getopt.h>
#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include "Int.h"
#include "IntGroup.h"
#include "Point.h"
#include "SECP256K1.h"
#include "ripemd160_avx512.h"
#include "sha256_avx512.h"

using namespace std;

// ================================================
// Funkcje pomocnicze
// ================================================

void initConsole() {
#ifdef _WIN32
  HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD mode = 0;
  GetConsoleMode(hConsole, &mode);
  SetConsoleMode(hConsole, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}

void clearTerminal() {
#ifdef _WIN32
  HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
  COORD coord = {0, 0};
  DWORD count;
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  GetConsoleScreenBufferInfo(hStdOut, &csbi);
  FillConsoleOutputCharacter(hStdOut, ' ', csbi.dwSize.X * csbi.dwSize.Y, coord, &count);
  SetConsoleCursorPosition(hStdOut, coord);
#else
  cout << "\033[2J\033[H";
#endif
  cout.flush();
}

void moveCursorTo(int x, int y) {
#ifdef _WIN32
  HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
  COORD coord = {(SHORT)x, (SHORT)y};
  SetConsoleCursorPosition(hStdOut, coord);
#else
  cout << "\033[" << y << ";" << x << "H";
#endif
  cout.flush();
}

// ================================================
// Stałe i zmienne globalne
// ================================================

int PUZZLE_NUM = 71;
int WORKERS = omp_get_num_procs();
int FLIP_COUNT = -1;
const __uint128_t REPORT_INTERVAL = 10000000;
static constexpr int POINTS_BATCH_SIZE = 512;
static constexpr int HASH_BATCH_SIZE = 32;

const unordered_map<int, tuple<int, string, string>> PUZZLE_DATA = {};

vector<unsigned char> TARGET_HASH160_RAW(20);
string TARGET_HASH160;
Int BASE_KEY;
atomic<bool> stop_event(false);
mutex result_mutex;
queue<tuple<string, __uint128_t, int>> results;

// ================================================
// AVXCounter (512-bit)
// ================================================

union AVXCounter {
  __m512i vec;
  uint64_t u64[8];
  __uint128_t u128[4];

  AVXCounter() : vec(_mm512_setzero_si512()) {}

  AVXCounter(__uint128_t value) { store(value); }

  void increment() {
    __m512i one = _mm512_set1_epi64(1);
    vec = _mm512_add_epi64(vec, one);
  }

  void store(__uint128_t value) {
    u128[0] = value;
    u128[1] = u128[2] = u128[3] = 0;
  }

  __uint128_t load() const { return (static_cast<__uint128_t>(u64[1]) << 64) | u64[0]; }

  bool operator<(const AVXCounter& other) const {
    for (int i = 7; i >= 0; i--) {
      if (u64[i] != other.u64[i]) return u64[i] < other.u64[i];
    }
    return false;
  }

  static AVXCounter div(const AVXCounter& num, uint64_t denom) {
    __uint128_t n = num.load();
    return AVXCounter(n / denom);
  }

  static uint64_t mod(const AVXCounter& num, uint64_t denom) { return num.load() % denom; }
};

static AVXCounter total_checked_avx;
__uint128_t total_combinations = 0;
vector<string> g_threadPrivateKeys;
mutex progress_mutex;

atomic<uint64_t> globalComparedCount(0);
atomic<uint64_t> localComparedCount(0);
double globalElapsedTime = 0.0;
double mkeysPerSec = 0.0;
chrono::time_point<chrono::high_resolution_clock> tStart;

// ================================================
// CombinationGenerator
// ================================================

class CombinationGenerator {
  int n, k;
  vector<int> current;

 public:
  CombinationGenerator(int n, int k) : n(n), k(k), current(k) {
    if (k > n) k = n;
    for (int i = 0; i < k; ++i) current[i] = i;
  }

  static __uint128_t combinations_count(int n, int k) {
    if (k > n) return 0;
    if (k * 2 > n) k = n - k;
    if (k == 0) return 1;

    __uint128_t result = n;
    for (int i = 2; i <= k; ++i) {
      result *= (n - k + i);
      result /= i;
    }
    return result;
  }

  const vector<int>& get() const { return current; }

  bool next() {
    int i = k - 1;
    while (i >= 0 && current[i] == n - k + i) --i;
    if (i < 0) return false;

    ++current[i];
    for (int j = i + 1; j < k; ++j) current[j] = current[j - 1] + 1;
    return true;
  }

  void unrank(__uint128_t rank) {
    current.resize(k);
    __uint128_t total = combinations_count(n, k);
    if (rank >= total) {
      current.clear();
      return;
    }

    __uint128_t remaining = rank;
    int a = n;
    int b = k;
    for (int i = 0; i < k; i++) {
      a = largest_a(a, b, remaining);
      current[i] = (n - 1) - a;
      remaining -= combinations_count(a, b);
      b--;
    }
  }

 private:
  int largest_a(int a, int b, __uint128_t x) const {
    while (a >= b && combinations_count(a, b) > x) a--;
    return a;
  }
};

// ================================================
// Funkcje haszujące
// ================================================

static void prepareShaBlock(const uint8_t* dataSrc, __uint128_t dataLen, uint8_t* outBlock) {
  fill_n(outBlock, 64, 0);
  memcpy(outBlock, dataSrc, dataLen);
  outBlock[dataLen] = 0x80;
  const uint32_t bitLen = static_cast<uint32_t>(dataLen * 8);
  copy(reinterpret_cast<const uint8_t*>(&bitLen) + 3, reinterpret_cast<const uint8_t*>(&bitLen) + 7,
       outBlock + 60);
}

static void prepareRipemdBlock(const uint8_t* dataSrc, uint8_t* outBlock) {
  fill_n(outBlock, 64, 0);
  memcpy(outBlock, dataSrc, 32);
  outBlock[32] = 0x80;
  const uint32_t bitLen = 256;
  copy(reinterpret_cast<const uint8_t*>(&bitLen) + 3, reinterpret_cast<const uint8_t*>(&bitLen) + 7,
       outBlock + 60);
}

static void computeHash160BatchAVX512(int numKeys, uint8_t pubKeys[][33],
                                      uint8_t hashResults[][20]) {
  alignas(64) array<array<uint8_t, 64>, HASH_BATCH_SIZE> shaInputs;
  alignas(64) array<array<uint8_t, 32>, HASH_BATCH_SIZE> shaOutputs;
  alignas(64) array<array<uint8_t, 64>, HASH_BATCH_SIZE> ripemdInputs;

  const uint8_t* inPtr[HASH_BATCH_SIZE];
  uint8_t* outPtr[HASH_BATCH_SIZE];

  for (int batch = 0; batch < (numKeys + HASH_BATCH_SIZE - 1) / HASH_BATCH_SIZE; batch++) {
    const int batchCount = min(HASH_BATCH_SIZE, numKeys - batch * HASH_BATCH_SIZE);

    for (int i = 0; i < batchCount; i++) {
      prepareShaBlock(pubKeys[batch * HASH_BATCH_SIZE + i], 33, shaInputs[i].data());
      inPtr[i] = shaInputs[i].data();
      outPtr[i] = shaOutputs[i].data();
    }

    sha256_avx512_32blocks(inPtr, outPtr);

    for (int i = 0; i < batchCount; i++) {
      prepareRipemdBlock(shaOutputs[i].data(), ripemdInputs[i].data());
      inPtr[i] = ripemdInputs[i].data();
      outPtr[i] = hashResults[batch * HASH_BATCH_SIZE + i];
    }

        ripemd160avx512::ripemd160avx512_64(
            const_cast<unsigned char*>(inPtr[0]), const_cast<unsigned char*>(inPtr[1]),
            ripemd160avx512::ripemd160avx512_64(
    const_cast<unsigned char*>(inPtr[0]), const_cast<unsigned char*>(inPtr[1]),
    const_cast<unsigned char*>(inPtr[2]), const_cast<unsigned char*>(inPtr[3]),
    const_cast<unsigned char*>(inPtr[4]), const_cast<unsigned char*>(inPtr[5]),
    const_cast<unsigned char*>(inPtr[6]), const_cast<unsigned char*>(inPtr[7]),
    const_cast<unsigned char*>(inPtr[8]), const_cast<unsigned char*>(inPtr[9]),
    const_cast<unsigned char*>(inPtr[10]), const_cast<unsigned char*>(inPtr[11]),
    const_cast<unsigned char*>(inPtr[12]), const_cast<unsigned char*>(inPtr[13]),
    const_cast<unsigned char*>(inPtr[14]), const_cast<unsigned char*>(inPtr[15]),
    outPtr[0], outPtr[1], outPtr[2], outPtr[3],
    outPtr[4], outPtr[5], outPtr[6], outPtr[7],
    outPtr[8], outPtr[9], outPtr[10], outPtr[11],
    outPtr[12], outPtr[13], outPtr[14], outPtr[15]
        );
  }
}

// ================================================
// Worker
// ================================================

void worker(Secp256K1* secp, int bit_length, int flip_count, int threadId, AVXCounter start,
            AVXCounter end) {
  const int fullBatchSize = 2 * POINTS_BATCH_SIZE;
  alignas(64) uint8_t localPubKeys[HASH_BATCH_SIZE][33];
  alignas(64) uint8_t localHashResults[HASH_BATCH_SIZE][20];
  alignas(64) int pointIndices[HASH_BATCH_SIZE];

  __m512i target = _mm512_loadu_si512(TARGET_HASH160_RAW.data());

  alignas(64) Point plusPoints[POINTS_BATCH_SIZE];
  alignas(64) Point minusPoints[POINTS_BATCH_SIZE];

  // Inicjalizacja punktów
  for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
    Int tmp;
    tmp.SetInt32(i);
    plusPoints[i] = secp->ComputePublicKey(&tmp);
    minusPoints[i] = plusPoints[i];
    minusPoints[i].y.ModNeg();
  }

  CombinationGenerator gen(bit_length, flip_count);
  gen.unrank(start.load());

  AVXCounter count;
  count.store(start.load());

  while (!stop_event.load() && count < end) {
    Int currentKey;
    currentKey.Set(&BASE_KEY);
    const vector<int>& flips = gen.get();

    // Aplikuj flips
    for (int pos : flips) {
      Int mask;
      mask.SetInt32(1);
      mask.ShiftL(pos);
      currentKey.Xor(&mask);
    }

    // Generuj klucze publiczne
    Point startPoint = secp->ComputePublicKey(&currentKey);
    Int startPointX = startPoint.x;
    Int startPointY = startPoint.y;

    // Obliczenia ECC
    Int deltaX[POINTS_BATCH_SIZE];
    IntGroup modGroup(POINTS_BATCH_SIZE);
    Int pointBatchX[fullBatchSize];
    Int pointBatchY[fullBatchSize];

#pragma omp simd
    for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
      deltaX[i].ModSub(&plusPoints[i].x, &startPointX);
    }

    modGroup.ModInv();

#pragma omp simd
    for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
      Int deltaY;
      deltaY.ModSub(&plusPoints[i].y, &startPointY);

      Int slope;
      slope.ModMulK1(&deltaY, &deltaX[i]);

      Int slopeSq;
      slopeSq.ModSquareK1(&slope);

      pointBatchX[i].ModSub(&slopeSq, &plusPoints[i].x);
      pointBatchX[i].ModAdd(&startPointX);

      Int diffX;
      diffX.ModSub(&startPointX, &pointBatchX[i]);
      diffX.ModMulK1(&slope);

      pointBatchY[i].ModSub(&startPointY, &diffX);
    }

#pragma omp simd
    for (int i = 0; i < POINTS_BATCH_SIZE; i++) {
      Int deltaY;
      deltaY.ModSub(&minusPoints[i].y, &startPointY);

      Int slope;
      slope.ModMulK1(&deltaY, &deltaX[i]);

      Int slopeSq;
      slopeSq.ModSquareK1(&slope);

      pointBatchX[POINTS_BATCH_SIZE + i].ModSub(&slopeSq, &minusPoints[i].x);
      pointBatchX[POINTS_BATCH_SIZE + i].ModAdd(&startPointX);

      Int diffX;
      diffX.ModSub(&startPointX, &pointBatchX[POINTS_BATCH_SIZE + i]);
      diffX.ModMulK1(&slope);

      pointBatchY[POINTS_BATCH_SIZE + i].ModSub(&startPointY, &diffX);
    }

    // Haszowanie i porównywanie
    computeHash160BatchAVX512(fullBatchSize, localPubKeys, localHashResults);

    // Sprawdź wyniki
    for (int i = 0; i < fullBatchSize; i++) {
      __m512i cand = _mm512_loadu_si512(localHashResults[i]);
      __mmask64 mask = _mm512_cmpeq_epi8_mask(cand, target);

      if (mask != 0) {
        Int foundKey;
        foundKey.Set(&currentKey);
        if (i < POINTS_BATCH_SIZE) {
          foundKey.Add(&Int(i));
        } else {
          foundKey.Sub(&Int(i - POINTS_BATCH_SIZE));
        }

        lock_guard<mutex> lock(result_mutex);
        results.push(make_tuple(foundKey.GetBase16(), total_checked_avx.load(), flip_count));
        stop_event.store(true);
        return;
      }
    }

    if (!gen.next()) break;
    count.increment();
  }
}

// ================================================
// Funkcja główna
// ================================================

int main(int argc, char* argv[]) {
  signal(SIGINT, [](int) { stop_event.store(true); });

  // Parsowanie argumentów
  int opt;
  while ((opt = getopt(argc, argv, "p:t:f:h")) != -1) {
    switch (opt) {
      case 'p':
        PUZZLE_NUM = atoi(optarg);
        break;
      case 't':
        WORKERS = atoi(optarg);
        break;
      case 'f':
        FLIP_COUNT = atoi(optarg);
        break;
      case 'h': {
        cout << "Usage: " << argv[0] << " [options]\n"
             << "Options:\n"
             << "  -p NUM  Puzzle number (1-256)\n"
             << "  -t NUM  Number of threads\n"
             << "  -f NUM  Bit flip count\n"
             << "  -h      Show this help\n";
        exit(0);
      }
      default:
        exit(1);
    }
  }

  // Sprawdź AVX-512
  if (!__builtin_cpu_supports("avx512f")) {
    cerr << "AVX-512 nie jest wspierany!\n";
    return 1;
  }

  // Inicjalizacja
  Secp256K1 secp;
  secp.Init();

  tStart = chrono::high_resolution_clock::now();
  total_combinations = CombinationGenerator::combinations_count(PUZZLE_NUM, FLIP_COUNT);

  // Uruchom wątki
  vector<thread> threads;
  AVXCounter total_combinations_avx;
  total_combinations_avx.store(total_combinations);

  AVXCounter comb_per_thread = AVXCounter::div(total_combinations_avx, WORKERS);
  uint64_t remainder = AVXCounter::mod(total_combinations_avx, WORKERS);

  for (int i = 0; i < WORKERS; i++) {
    AVXCounter start, end;
    start.store(AVXCounter::mul(i, comb_per_thread.load()).load() +
                min(static_cast<uint64_t>(i), remainder));
    end.store(start.load() + comb_per_thread.load() + (i < remainder ? 1 : 0));

    threads.emplace_back(worker, &secp, PUZZLE_NUM, FLIP_COUNT, i, start, end);
  }

  // Monitoruj postęp
  while (!stop_event) {
    this_thread::sleep_for(chrono::seconds(1));

    lock_guard<mutex> lock(progress_mutex);
    __uint128_t current_total = total_checked_avx.load();
    globalElapsedTime =
        chrono::duration<double>(chrono::high_resolution_clock::now() - tStart).count();

    cout << "Progress: " << fixed << setprecision(6)
         << (double)current_total / total_combinations * 100.0 << "%\n"
         << "Speed: " << fixed << setprecision(2) << (current_total / globalElapsedTime) / 1e6
         << " Mkeys/s\n";
  }

  if (!results.empty()) {
    auto [hexKey, count, flips] = results.front();

    cout << "\n\n=======================================\n"
         << "Private key found: " << hexKey << "\n"
         << "Bit flips: " << flips << "\n"
         << "Total checked: " << to_string_128(count) << "\n"
         << "Time: " << formatElapsedTime(globalElapsedTime) << "\n"
         << "=======================================\n";

    ofstream fout("solution.txt");
    if (fout) {
      fout << hexKey;
      cout << "Saved to solution.txt\n";
    }

    // Zatrzymaj wszystkie wątki
    stop_event.store(true);
    for (auto& t : threads)
      if (t.joinable()) t.join();

    return 0;
  }
