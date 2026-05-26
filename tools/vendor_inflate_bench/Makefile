# Standalone Makefile for the vendor C++ inflate bench (Phase 1.4).
#
# Builds main.cpp against rapidgzip vendor headers, links against the
# static libs vendor's own CMake build produces (librpmalloc.a +
# libzlibstatic.a + libisal.a). Assumes vendor build tree is at
# ../../vendor/rapidgzip/librapidarchive/build/ — run vendor build
# first (`cd vendor/rapidgzip/librapidarchive && mkdir -p build && cd
# build && cmake .. && cmake --build . --target rapidgzip`) if it
# doesn't exist.

ROOT := $(realpath ../..)
VENDOR := $(ROOT)/vendor/rapidgzip/librapidarchive
BUILD := $(VENDOR)/build

INCLUDES := \
	-I$(VENDOR)/src \
	-I$(VENDOR)/src/external \
	-I$(VENDOR)/src/external/cxxopts/include \
	-I$(BUILD)/_deps/isal_project-build \
	-I$(BUILD)/_deps/isal_project-src/include

LIBS := \
	$(BUILD)/src/librpmalloc.a \
	$(BUILD)/src/libzlibstatic.a \
	$(BUILD)/_deps/isal_project-build/libisal.a

CXX := g++
CXXFLAGS := -O3 -std=c++17 -DNDEBUG -march=x86-64-v3 -pthread $(INCLUDES)
LDFLAGS := -pthread

vendor_inflate_bench: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp $(LIBS) $(LDFLAGS) -o vendor_inflate_bench

run: vendor_inflate_bench
	./vendor_inflate_bench $(ROOT)/corpus/silesia_blocks

clean:
	rm -f vendor_inflate_bench

.PHONY: run clean
