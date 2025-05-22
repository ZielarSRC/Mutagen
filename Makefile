# Detect OS
UNAME_S := $(shell uname -s)

# Enable static linking by default
STATIC_LINKING = yes

# Compiler settings based on OS
ifeq ($(UNAME_S),Linux)
# Linux settings

# Compiler
CXX = g++

# Compiler flags dla AVX-512
CXXFLAGS = -m64 -std=c++20 -Ofast -Wall -Wextra \
           -Wno-write-strings -Wno-unused-variable -Wno-deprecated-copy \
           -Wno-unused-parameter -Wno-sign-compare -Wno-strict-aliasing \
           -Wno-unused-but-set-variable \
           -funroll-loops -ftree-vectorize -fstrict-aliasing \
           -fno-semantic-interposition -fvect-cost-model=unlimited \
           -fno-trapping-math -fipa-ra -flto -fassociative-math \
           -fopenmp -mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq \
           -mbmi2 -madx -fwrapv

# Source files z AVX-512
SRCS = mutagen.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp \
       Point.cpp ripemd160_avx512.cpp sha256_avx512.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = mutagen

# Linkowanie
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -static
	rm -f $(OBJS) && chmod +x $(TARGET)

# Kompilacja
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Czyszczenie
clean:
	@echo "Cleaning..."
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean 

else
# Windows settings (MinGW-w64)

# Compiler
CXX = g++

# Sprawdzenie kompilatora
CHECK_COMPILER := $(shell which $(CXX))

# Dodanie ścieżki MSYS
ifeq ($(CHECK_COMPILER),)
  $(info Compiler not found. Adding MSYS path to the environment...)
  SHELL := powershell
  PATH := C:\msys64\mingw64\bin;$(PATH)
endif

# Flagi dla AVX-512
CXXFLAGS = -m64 -std=c++20 -Ofast -Wall -Wextra \
           -Wno-write-strings -Wno-unused-variable -Wno-deprecated-copy \
           -Wno-unused-parameter -Wno-sign-compare -Wno-strict-aliasing \
           -Wno-unused-but-set-variable -funroll-loops -ftree-vectorize \
           -fstrict-aliasing -fno-semantic-interposition \
           -fvect-cost-model=unlimited -fno-trapping-math -fipa-ra \
           -fassociative-math -fopenmp -mavx512f -mavx512cd -mavx512vl \
           -mavx512bw -mavx512dq -mbmi2 -madx -fwrapv

# Static linking
ifeq ($(STATIC_LINKING), yes)
    CXXFLAGS += -static
else
    $(info Dynamic linking will be used. Ensure required DLLs are distributed)
endif

# Source files
SRCS = mutagen.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp \
       Point.cpp ripemd160_avx512.cpp sha256_avx512.cpp

OBJS = $(SRCS:.cpp=.o)
TARGET = mutagen.exe

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
	del /q $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo Cleaning...
	del /q $(OBJS) $(TARGET)

.PHONY: all clean
endif
