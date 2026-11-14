#!/bin/sh
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if command -v nvcc >/dev/null 2>&1; then
 nvcc -O3 -std=c++17 -arch=sm_120 -c "$DIR/gpu.cu" -o "$DIR/gpu.o"
 c++ -fopenmp -O3 -march=native -std=c++17 -DPRISM_CUDA -c "$DIR/main.cpp" -o "$DIR/main.o"
 c++ -fopenmp -O3 -o "$DIR/prismalign" "$DIR/main.o" "$DIR/gpu.o" -lcudart
else
 c++ -fopenmp -O3 -march=native -std=c++17 -o "$DIR/prismalign" "$DIR/main.cpp"
fi
