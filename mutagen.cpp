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
// Narzędzia do wyświetlania dużych liczb i czasu
// ================================================

std::string to_string_128(__uint128_t val) {
  if (val == 0) return "0";
  std::string ret;
  while (val) {
    ret.push_back('0' + (val % 10));
    val /= 10;
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

std::string formatElapsedTime(double seconds) {
  int h = int(seconds) / 3600;
  int m = (int(seconds) % 3600) / 60;
  double s = std::fmod(seconds, 60.0);
  std::ostringstream oss;
  if (h > 0) oss << h << "h ";
  if (h > 0 || m > 0) oss << m << "m ";
  oss << std::fixed << std::setprecision(2) << s << "s";
  return oss.str();
}

// ================================================
// Stałe i zmienne globalne - WYDAJNOŚĆ MAKSYMALNA
// ================================================

int PUZZLE_NUM = 20;
int WORKERS = 1;
int FLIP_COUNT = -1;
const __uint128_t REPORT_INTERVAL = 1000000;
// AVX-512 MEGA BATCHES dla Xeon Platinum 8488C!
static constexpr int POINTS_BATCH_SIZE = 512;  // MEGA BATCH AVX-512!
static constexpr int HASH_BATCH_SIZE = 64;     // MEGA BATCH AVX-512!

const unordered_map<int, tuple<int, string, string>> PUZZLE_DATA = {
    {20, {8, "b907c3a2a3b27789dfb509b730dd47703c272868", "357535"}},
    {21, {9, "29a78213caa9eea824acf08022ab9dfc83414f56", "863317"}},
    {22, {11, "7ff45303774ef7a52fffd8011981034b258cb86b", "1811764"}},
    {23, {12, "d0a79df189fe1ad5c306cc70497b358415da579e", "3007503"}},
    {24, {9, "0959e80121f36aea13b3bad361c15dac26189e2f", "5598802"}},
    {25, {12, "2f396b29b27324300d0c59b17c3abc1835bd3dbb", "14428676"}},
    {26, {14, "bfebb73562d4541b32a02ba664d140b5a574792f", "33185509"}},
    {27, {13, "0c7aaf6caa7e5424b63d317f0f8f1f9fa40d5560", "54538862"}},
    {28, {16, "1306b9e4ff56513a476841bac7ba48d69516b1da", "111949941"}},
    {29, {18, "5a416cc9148f4a377b672c8ae5d3287adaafadec", "227634408"}},
    {30, {16, "d39c4704664e1deb76c9331e637564c257d68a08", "400708894"}},
    {31, {13, "d805f6f251f7479ebd853b3d0f4b9b2656d92f1d", "1033162084"}},
    {32, {14, "9e42601eeaedc244e15f17375adb0e2cd08efdc9", "2102388551"}},
    {33, {15, "4e15e5189752d1eaf444dfd6bff399feb0443977", "3093472814"}},
    {34, {16, "f6d67d7983bf70450f295c9cb828daab265f1bfa", "7137437912"}},
    {35, {19, "f6d8ce225ffbdecec170f8298c3fc28ae686df25", "14133072157"}},
    {36, {14, "74b1e012be1521e5d8d75e745a26ced845ea3d37", "20112871792"}},
    {37, {23, "28c30fb9118ed1da72e7c4f89c0164756e8a021d", "42387769980"}},
    {38, {21, "b190e2d40cfdeee2cee072954a2be89e7ba39364", "100251560595"}},
    {39, {23, "0b304f2a79a027270276533fe1ed4eff30910876", "146971536592"}},
    {40, {20, "95a156cd21b4a69de969eb6716864f4c8b82a82a", "323724968937"}},
    {41, {25, "d1562eb37357f9e6fc41cb2359f4d3eda4032329", "1003651412950"}},
    {42, {24, "8efb85f9c5b5db2d55973a04128dc7510075ae23", "1458252205147"}},
    {43, {19, "f92044c7924e5525c61207972c253c9fc9f086f7", "2895374552463"}},
    {44, {24, "80df54e1f612f2fc5bdc05c9d21a83aa8d20791e", "7409811047825"}},
    {45, {21, "f0225bfc68a6e17e87cd8b5e60ae3be18f120753", "15404761757071"}},
    {46, {24, "9a012260d01c5113df66c8a8438c9f7a1e3d5dac", "19996463086597"}},
    {47, {27, "f828005d41b0f4fed4c8dca3b06011072cfb07d4", "51408670348612"}},
    {48, {21, "8661cb56d9df0a61f01328b55af7e56a3fe7a2b2", "119666659114170"}},
    {49, {30, "0d2f533966c6578e1111978ca698f8add7fffdf3", "191206974700443"}},
    {50, {29, "de081b76f840e462fa2cdf360173dfaf4a976a47", "409118905032525"}},
    {51, {25, "ef6419cffd7fad7027994354eb8efae223c2dbe7", "611140496167764"}},
    {52, {27, "36af659edbe94453f6344e920d143f1778653ae7", "2058769515153876"}},
    {53, {26, "2f4870ef54fa4b048c1365d42594cc7d3d269551", "4216495639600700"}},
    {54, {30, "cb66763cf7fde659869ae7f06884d9a0f879a092", "6763683971478124"}},
    {55, {31, "db53d9bbd1f3a83b094eeca7dd970bd85b492fa2", "9974455244496707"}},
    {56, {31, "48214c5969ae9f43f75070cea1e2cb41d5bdcccd", "30045390491869460"}},
    {57, {33, "328660ef43f66abe2653fa178452a5dfc594c2a1", "44218742292676575"}},
    {58, {28, "8c2a6071f89c90c4dab5ab295d7729d1b54ea60f", "138245758910846492"}},
    {59, {30, "b14ed3146f5b2c9bde1703deae9ef33af8110210", "199976667976342049"}},
    {60, {31, "cdf8e5c7503a9d22642e3ecfc87817672787b9c5", "525070384258266191"}},
    {61, {25, "68133e19b2dfb9034edf9830a200cfdf38c90cbd", "1135041350219496382"}},
    {62, {35, "e26646db84b0602f32b34b5a62ca3cae1f91b779", "1425787542618654982"}},
    {63, {34, "ef58afb697b094423ce90721fbb19a359ef7c50e", "3908372542507822062"}},
    {64, {34, "3ee4133d991f52fdf6a25c9834e0745ac74248a4", "8993229949524469768"}},
    {65, {37, "52e763a7ddc1aa4fa811578c491c1bc7fd570137", "17799667357578236628"}},
    {66, {35, "20d45a6a762535700ce9e0b216e31994335db8a5", "30568377312064202855"}},
    {67, {31, "739437bb3dd6d1983e66629c5f08c70e52769371", "46346217550346335726"}},
    {68, {42, "e0b8a2baee1b77fc703455f39d51477451fc8cfc", "132656943602386256302"}},
    {69, {34, "61eb8a50c86b0584bb727dd65bed8d2400d6d5aa", "219898266213316039825"}},
    {70, {29, "5db8cda53a6a002db10365967d7f85d19e171b10", "297274491920375905804"}},
    {71, {29, "f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8", "970436974005023690481"}}};

vector<unsigned char> TARGET_HASH160_RAW(20);
string TARGET_HASH160;
Int BASE_KEY;
atomic<bool> stop_event(false);
mutex result_mutex;
queue<tuple<string, __uint128_t, int>> results;

// ================================================
// HARDCORE AVX-512 Counter dla MAKSYMALNEJ WYDAJNOŚCI!
// ================================================

union AVXCounter {
  __m512i vec512;  // PEŁNY AVX-512!
  __m256i vec256[2];
  uint64_t u64[8];
  __uint128_t u128[4];

  AVXCounter() : vec512(_mm512_setzero_si512()) {}

  AVXCounter(__uint128_t value) { store(value); }

  // MEGA WYDAJNE INKREMENTOWANIE AVX-512!
  void increment() {
    __m512i one = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, 1);
    vec512 = _mm512_add_epi64(vec512, one);
  }

  // ULTRA SZYBKIE batch increment dla MEGA WYDAJNOŚCI!
  void increment_batch(uint64_t batch_size) {
    __m512i batch = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, batch_size);
    vec512 = _mm512_add_epi64(vec512, batch);
  }

  void store(__uint128_t value) {
    u64[0] = static_cast<uint64_t>(value);
    u64[1] = static_cast<uint64_t>(value >> 64);
    u64[2] = 0;
    u64[3] = 0;
    u64[4] = 0;
    u64[5] = 0;
    u64[6] = 0;
    u64[7] = 0;
  }

  __uint128_t load() const { return (static_cast<__uint128_t>(u64[1]) << 64) | u64[0]; }

  bool operator<(const AVXCounter& other) const {
    if (u64[1] != other.u64[1]) return u64[1] < other.u64[1];
    return u64[0] < other.u64[0];
  }

  static AVXCounter div(const AVXCounter& num, uint64_t denom) {
    __uint128_t n = num.load();
    return AVXCounter(n / denom);
  }

  static uint64_t mod(const AVXCounter& num, uint64_t denom) { return num.load() % denom; }

  static AVXCounter mul(uint64_t a, uint64_t b) {
    return AVXCounter(static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b));
  }
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
// WYDAJNY CombinationGenerator
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

    // POPRAWKA: Usunięto nieużywaną zmienną remaining
    int a = n;
    int b = k;
    __uint128_t x = (total - 1) - rank;
    for (int i = 0; i < k; i++) {
      a = largest_a_where_comb_a_b_le_x(a, b, x);
      current[i] = (n - 1) - a;
      x -= combinations_count(a, b);
      b--;
    }
  }

 private:
  int largest_a_where_comb_a_b_le_x(int a, int b, __uint128_t x) const {
    while (a >= b && combinations_count(a, b) > x) a--;
    return a;
  }
};

// ================================================
// HARDCORE MEGA WYDAJNY WORKER dla Xeon Platinum 8488C!
// ================================================

void worker(Secp256K1* secp, int bit_length, int flip_count, int threadId, AVXCounter start,
            AVXCounter end) {
  cout << "🚀 HARDCORE Thread " << threadId << " starting (range: " << to_string_128(start.load())
       << " to " << to_string_128(end.load()) << ")\n";

  // MEGA BATCH ARRAYS dla AVX-512 WYDAJNOŚCI!
  alignas(64) uint8_t hash160_batch[HASH_BATCH_SIZE][20];
  alignas(64) Point pubPoints_batch[POINTS_BATCH_SIZE];
  alignas(64) Int keys_batch[POINTS_BATCH_SIZE];

  CombinationGenerator gen(bit_length, flip_count);
  gen.unrank(start.load());

  AVXCounter count;
  count.store(start.load());

  uint64_t localIterations = 0;
  uint64_t batch_processed = 0;

  while (!stop_event.load() && count < end) {
    // MEGA BATCH PROCESSING dla MAKSYMALNEJ WYDAJNOŚCI!
    int batch_size = min(POINTS_BATCH_SIZE, static_cast<int>(end.load() - count.load()));
    if (batch_size <= 0) break;

    // Przygotuj MEGA BATCH kluczy
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      keys_batch[batch_idx].Set(&BASE_KEY);
      const vector<int>& flips = gen.get();

      // ULTRA SZYBKA aplikacja flips z loop unrolling
      for (size_t flip_idx = 0; flip_idx < flips.size(); flip_idx++) {
        Int mask;
        mask.SetInt32(1);
        mask.ShiftL(flips[flip_idx]);
        keys_batch[batch_idx].Xor(&mask);
      }

      if (!gen.next() && batch_idx < batch_size - 1) {
        batch_size = batch_idx + 1;
        break;
      }
    }

    // MEGA BATCH obliczenia public keys z SIMD!
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      pubPoints_batch[batch_idx] = secp->ComputePublicKey(&keys_batch[batch_idx]);
    }

    // MEGA BATCH hash160 z AVX-512!
    if (batch_size >= 16) {
      // PEŁNY AVX-512 BATCH!
      for (int batch_start = 0; batch_start + 16 <= batch_size; batch_start += 16) {
        Point* pubkey_ptrs[16];
        uint8_t* hash_ptrs[16];

        for (int i = 0; i < 16; i++) {
          pubkey_ptrs[i] = &pubPoints_batch[batch_start + i];
          hash_ptrs[i] = hash160_batch[batch_start + i];
        }

        secp->GetHash160_Batch16(P2PKH, true, pubkey_ptrs, hash_ptrs);
      }

      // Pozostałe pojedynczo
      for (int remaining = (batch_size / 16) * 16; remaining < batch_size; remaining++) {
        secp->GetHash160(P2PKH, true, pubPoints_batch[remaining], hash160_batch[remaining]);
      }
    } else {
      // Mniejsze batche
      for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        secp->GetHash160(P2PKH, true, pubPoints_batch[batch_idx], hash160_batch[batch_idx]);
      }
    }

    // ULTRA SZYBKA weryfikacja z SIMD porównaniami
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      // AVX-512 porównanie w jednej instrukcji!
      __m512i target =
          _mm512_loadu_si512(reinterpret_cast<const __m512i*>(TARGET_HASH160_RAW.data()));
      __m512i hash = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(hash160_batch[batch_idx]));

      // Porównaj pierwsze 20 bajtów
      __mmask64 mask = _mm512_cmpeq_epi8_mask(target, hash);

      // Sprawdź czy pierwsze 20 bitów to 1 (czyli match)
      if ((mask & 0xFFFFF) == 0xFFFFF) {
        lock_guard<mutex> lock(result_mutex);
        results.push(
            make_tuple(keys_batch[batch_idx].GetBase16(), total_checked_avx.load(), flip_count));
        stop_event.store(true);
        cout << "🎉 MEGA HARDCORE Thread " << threadId << " found solution!\n";
        return;
      }
    }

    localIterations += batch_size;
    batch_processed++;
    count.increment_batch(batch_size);
    total_checked_avx.increment_batch(batch_size);
    localComparedCount += batch_size;

    // Raportowanie co MEGA_BATCH
    if (batch_processed % 1000 == 0) {
      cout << "🚀 MEGA Thread " << threadId << " processed " << batch_processed << " MEGA batches ("
           << localIterations << " keys)\n";
    }
  }

  cout << "🏁 HARDCORE Thread " << threadId << " finished (" << localIterations << " iterations, "
       << batch_processed << " batches)\n";
}

// ================================================
// MAIN - PEŁNA WYDAJNOŚĆ!
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
             << "  -p NUM  Puzzle number (20-71)\n"
             << "  -t NUM  Number of threads\n"
             << "  -f NUM  Bit flip count\n"
             << "  -h      Show this help\n";
        exit(0);
      }
      default:
        exit(1);
    }
  }

  // MEGA SPRAWDZENIE AVX-512!
  if (!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512dq") ||
      !__builtin_cpu_supports("avx512cd") || !__builtin_cpu_supports("avx512bw") ||
      !__builtin_cpu_supports("avx512vl")) {
    cerr << "❌ PEŁNY AVX-512 nie jest wspierany na tym CPU!\n";
    return 1;
  }

  cout << "🚀 HARDCORE AVX-512 wsparcie potwierdzone - FULL POWER MODE!\n";

  cout << "🔧 MEGA Inicjalizacja SECP256K1...\n";
  Secp256K1 secp;

  try {
    secp.Init();
    cout << "✅ SECP256K1 zainicjalizowane - TURBO MODE!\n";
  } catch (const exception& e) {
    cerr << "❌ Błąd podczas inicjalizacji SECP256K1: " << e.what() << "\n";
    return 1;
  }

  // Test MEGA WYDAJNOŚCI
  cout << "🔧 Test MEGA WYDAJNOŚCI ECC...\n";
  try {
    Int testKey;
    testKey.SetInt32(1);
    Point testPoint = secp.ComputePublicKey(&testKey);
    cout << "✅ Test ECC - TURBO SPEED!\n";
  } catch (...) {
    cerr << "❌ Błąd podczas testu ECC\n";
    return 1;
  }

  // Pobierz dane puzzla
  auto puzzle_it = PUZZLE_DATA.find(PUZZLE_NUM);
  if (puzzle_it == PUZZLE_DATA.end()) {
    cerr << "❌ Invalid puzzle number (20-71)\n";
    return 1;
  }

  auto [DEFAULT_FLIP_COUNT, TARGET_HASH160_HEX, PRIVATE_KEY_DECIMAL] = puzzle_it->second;

  if (FLIP_COUNT == -1) {
    FLIP_COUNT = DEFAULT_FLIP_COUNT;
  }

  TARGET_HASH160 = TARGET_HASH160_HEX;

  cout << "🔧 MEGA Inicjalizacja target hash...\n";
  for (int i = 0; i < 20; i++) {
    TARGET_HASH160_RAW[i] = stoul(TARGET_HASH160.substr(i * 2, 2), nullptr, 16);
  }

  cout << "🔧 MEGA Inicjalizacja base key...\n";
  BASE_KEY.SetBase10(const_cast<char*>(PRIVATE_KEY_DECIMAL.c_str()));

  Int testKey;
  testKey.SetBase10(const_cast<char*>(PRIVATE_KEY_DECIMAL.c_str()));
  if (!testKey.IsEqual(&BASE_KEY)) {
    cerr << "❌ Base key initialization failed!\n";
    return 1;
  }

  cout << "✅ Base key zainicjalizowany - MEGA POWER!\n";

  tStart = chrono::high_resolution_clock::now();
  total_combinations = CombinationGenerator::combinations_count(PUZZLE_NUM, FLIP_COUNT);

  string paddedKey = BASE_KEY.GetBase16();
  size_t firstNonZero = paddedKey.find_first_not_of('0');

  if (string::npos == firstNonZero) {
    paddedKey = "0";
  } else {
    paddedKey = paddedKey.substr(firstNonZero);
  }

  paddedKey = "0x" + paddedKey;

  clearTerminal();
  cout << "=======================================\n";
  cout << "== 🚀 MUTAGEN HARDCORE AVX-512 TURBO ==\n";
  cout << "=======================================\n";
  cout << "🎯 Starting puzzle: " << PUZZLE_NUM << " (" << PUZZLE_NUM << "-bit)\n";
  cout << "🔍 Target HASH160: " << TARGET_HASH160 << "\n";
  cout << "🗝️  Base Key: " << paddedKey << "\n";
  cout << "🔄 Flip count: " << FLIP_COUNT << "\n";
  cout << "📊 Total combinations: " << to_string_128(total_combinations) << "\n";
  cout << "🚀 Using: " << WORKERS << " HARDCORE threads\n";
  cout << "💥 MEGA BATCH SIZE: " << POINTS_BATCH_SIZE << " (AVX-512 TURBO!)\n\n";

  g_threadPrivateKeys.resize(WORKERS, "0");

  // MEGA LAUNCH threads!
  vector<thread> threads;
  AVXCounter total_combinations_avx;
  total_combinations_avx.store(total_combinations);

  AVXCounter comb_per_thread = AVXCounter::div(total_combinations_avx, WORKERS);
  uint64_t remainder = AVXCounter::mod(total_combinations_avx, WORKERS);

  cout << "🚀 LAUNCHING " << WORKERS << " HARDCORE worker threads...\n\n";

  for (int i = 0; i < WORKERS; i++) {
    AVXCounter start, end;

    AVXCounter base = AVXCounter::mul(i, comb_per_thread.load());
    uint64_t extra = min(static_cast<uint64_t>(i), remainder);
    start.store(base.load() + extra);

    // POPRAWKA: rzutowanie na uint64_t dla porównania
    end.store(start.load() + comb_per_thread.load() +
              (static_cast<uint64_t>(i) < remainder ? 1 : 0));
    threads.emplace_back(worker, &secp, PUZZLE_NUM, FLIP_COUNT, i, start, end);
  }

  // MEGA monitoring
  int monitoring_cycles = 0;
  while (!stop_event.load()) {
    this_thread::sleep_for(chrono::seconds(5));
    monitoring_cycles++;

    cout << "⚡ HARDCORE Monitoring cycle " << monitoring_cycles << "\n";
    cout << "🚀 Total checked: " << to_string_128(total_checked_avx.load()) << "\n";
    cout << "💥 Local compared: " << localComparedCount.load() << "\n";

    if (total_checked_avx.load() >= total_combinations) {
      stop_event.store(true);
      break;
    }
  }

  cout << "🏁 HARDCORE - Waiting for threads to finish...\n";
  for (auto& t : threads) {
    if (t.joinable()) t.join();
  }

  if (!results.empty()) {
    auto [hexKey, count, flips] = results.front();
    globalElapsedTime =
        chrono::duration<double>(chrono::high_resolution_clock::now() - tStart).count();

    cout << "\n\n🎉 =======================================\n"
         << "🏆 ====== HARDCORE SOLUTION FOUND! ======\n"
         << "🎉 =======================================\n"
         << "🗝️  Private key: " << hexKey << "\n"
         << "🔄 Bit flips: " << flips << "\n"
         << "📊 Total checked: " << to_string_128(count) << "\n"
         << "⏱️  Time: " << formatElapsedTime(globalElapsedTime) << "\n";

    ofstream fout("solution.txt");
    if (fout) {
      fout << hexKey;
      cout << "💾 Saved to solution.txt\n";
    }
  } else {
    globalElapsedTime =
        chrono::duration<double>(chrono::high_resolution_clock::now() - tStart).count();
    cout << "\n\n❌ No solution found\n";
    cout << "📊 Total checked: " << to_string_128(total_checked_avx.load()) << "\n";
    cout << "⏱️  Time: " << formatElapsedTime(globalElapsedTime) << "\n";
  }

  return 0;
}
