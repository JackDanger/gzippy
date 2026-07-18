cmake_minimum_required(VERSION 3.16)

# Standalone CMakeLists for the vendor bench. Run from the vendor build
# tree so we pick up its target definitions (librapidgzip + headers).
#
# Wiring is in tools/vendor_inflate_bench/Makefile and the top-level
# Makefile's `vendor-bench` target — they invoke vendor's cmake from
# within vendor/rapidgzip/librapidarchive/build/ and add this directory
# as a subdirectory via -DLIBRAPIDARCHIVE_VENDOR_BENCH=ON.

add_executable(vendor_inflate_bench)
target_sources(vendor_inflate_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
target_link_libraries(vendor_inflate_bench PRIVATE librapidgzip)
target_compile_features(vendor_inflate_bench PRIVATE cxx_std_17)
