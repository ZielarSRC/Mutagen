CXX = g++
CXXFLAGS = -O3 -march=sapphirerapids -mtune=sapphirerapids -mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl -fopenmp -funroll-loops -ffast-math -Wall

OBJS = main.o SECP256K1.o Int.o Point.o IntGroup.o IntMod.o sha256_avx512.o ripemd160_avx512.o

all: mutagen

mutagen: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o mutagen
