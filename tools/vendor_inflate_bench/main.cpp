// Phase 1.4 vendor C++ bench harness (see plans/inner-loop-execution.md).
//
// Reads the same corpus files as benches/inflate_block.rs (GZBLK01
// format documented in tools/extract_blocks/src/main.rs), runs vendor's
// rapidgzip::deflate::Block::read on each block, prints MB/s in
// JSON-line format.
//
// No FFI, no link against Rust. Build via the Makefile target
// `make vendor-bench`, which sets -I paths to vendor headers and links
// against the static librapidgzip.a built in the vendor CMake tree.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

// Vendor headers
#include <rapidgzip/gzip/deflate.hpp>
#include <gzip/BitReader.hpp>

namespace fs = std::filesystem;

struct CorpusBlock {
    std::string file_name;
    uint32_t block_idx;
    uint64_t bit_offset_in_compressed;
    uint64_t compressed_bit_len;
    std::vector<uint8_t> compressed;
    std::vector<uint8_t> predecessor;
    std::vector<uint8_t> decoded_expected;
    std::vector<uint8_t> literal_cl;
    std::vector<uint8_t> distance_cl;
    uint8_t max_litlen_code_len;
    uint8_t max_dist_code_len;
};

static uint32_t read_u32_le(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
static uint64_t read_u64_le(const uint8_t* p) {
    return uint64_t(read_u32_le(p)) | (uint64_t(read_u32_le(p + 4)) << 32);
}
static uint16_t read_u16_le(const uint8_t* p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

static CorpusBlock load_block(const fs::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "open: " << path << "\n";
        std::exit(2);
    }
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size);

    if (size < 64 || std::memcmp(data.data(), "GZBLK01\0", 8) != 0) {
        std::cerr << "bad magic: " << path << "\n";
        std::exit(2);
    }

    CorpusBlock b;
    b.file_name = path.filename().string();
    b.block_idx = read_u32_le(&data[8]);
    // bytes 12-15: btype (skip; all dynamic by construction)
    b.bit_offset_in_compressed = read_u64_le(&data[16]);
    b.compressed_bit_len = read_u64_le(&data[24]);
    uint64_t compressed_byte_len = read_u64_le(&data[32]);
    uint32_t decoded_len = read_u32_le(&data[40]);
    uint32_t predecessor_len = read_u32_le(&data[44]);
    uint16_t literal_code_count = read_u16_le(&data[48]);
    uint16_t distance_code_count = read_u16_le(&data[50]);
    b.max_litlen_code_len = data[52];
    b.max_dist_code_len = data[53];
    // skip marker_count_during_decode (54-55), crc32 (56-59), reserved (60-63)

    size_t cursor = 64;
    b.predecessor.assign(data.begin() + cursor, data.begin() + cursor + predecessor_len);
    cursor += predecessor_len;
    b.compressed.assign(data.begin() + cursor, data.begin() + cursor + compressed_byte_len);
    cursor += compressed_byte_len;
    b.decoded_expected.assign(data.begin() + cursor, data.begin() + cursor + decoded_len);
    cursor += decoded_len;
    b.literal_cl.assign(data.begin() + cursor, data.begin() + cursor + literal_code_count);
    cursor += literal_code_count;
    b.distance_cl.assign(data.begin() + cursor, data.begin() + cursor + distance_code_count);

    return b;
}

// Decode one block via vendor's rapidgzip::deflate::Block.
// Returns the decoded body byte count.
template<bool IsBootstrap>
static size_t decode_one(const CorpusBlock& blk) {
    using namespace rapidgzip;

    // BitReader needs a "file reader" view of the compressed bytes.
    // Vendor's BufferViewFileReader serves; available in gzip/BitReader.hpp area.
    auto reader = std::make_unique<BufferViewFileReader>(
        reinterpret_cast<const char*>(blk.compressed.data()),
        blk.compressed.size());
    gzip::BitReader bitReader(std::move(reader));
    bitReader.seek(static_cast<long long int>(blk.bit_offset_in_compressed));

    deflate::Block block;

    if constexpr (!IsBootstrap) {
        // Fast path: seed window with predecessor. setInitialWindow flips
        // containsMarkerBytes off so back-refs resolve to clean bytes.
        // Vendor signature: setInitialWindow(VectorView<uint8_t>)
        VectorView<uint8_t> window(
            blk.predecessor.data(),
            blk.predecessor.size());
        const auto err = block.setInitialWindow(window);
        if (err != Error::NONE) {
            std::cerr << "setInitialWindow err on " << blk.file_name << ": "
                      << static_cast<int>(err) << "\n";
            std::exit(3);
        }
    }
    // For bootstrap: no setInitialWindow; ring stays in marker mode.

    const auto err = block.readHeader(bitReader);
    if (err != Error::NONE) {
        std::cerr << "readHeader err on " << blk.file_name << ": "
                  << static_cast<int>(err) << "\n";
        std::exit(3);
    }

    size_t total_decoded = 0;
    while (!block.eob()) {
        auto result = block.read(bitReader, std::numeric_limits<size_t>::max());
        if (result.second != Error::NONE) {
            std::cerr << "read err on " << blk.file_name << ": "
                      << static_cast<int>(result.second) << "\n";
            std::exit(3);
        }
        total_decoded += result.first.size();
    }

    return total_decoded;
}

template<bool IsBootstrap>
static void run_bench(const std::vector<CorpusBlock>& corpus, const char* label) {
    // Warm up
    for (const auto& blk : corpus) {
        decode_one<IsBootstrap>(blk);
    }

    // Measure
    constexpr int N_ITERS = 200;
    uint64_t total_decoded_bytes = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N_ITERS; i++) {
        for (const auto& blk : corpus) {
            total_decoded_bytes += decode_one<IsBootstrap>(blk);
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double mb_s = (double)total_decoded_bytes / 1.048576e6 / secs;
    double per_iter_us = secs * 1e6 / N_ITERS;

    std::printf("{\"label\":\"%s\",\"iters\":%d,\"total_decoded_bytes\":%llu,"
                "\"total_seconds\":%.6f,\"per_iter_us\":%.2f,\"mib_per_sec\":%.2f}\n",
                label, N_ITERS,
                (unsigned long long)total_decoded_bytes, secs, per_iter_us, mb_s);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: vendor_inflate_bench <corpus_dir>\n";
        return 2;
    }
    fs::path corpus_dir = argv[1];

    std::vector<CorpusBlock> corpus;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(corpus_dir)) {
        if (e.path().extension() == ".bin") {
            files.push_back(e.path());
        }
    }
    std::sort(files.begin(), files.end());
    for (const auto& p : files) {
        corpus.push_back(load_block(p));
    }
    std::fprintf(stderr, "loaded %zu corpus blocks from %s\n",
                 corpus.size(), corpus_dir.c_str());

    run_bench<false>(corpus, "inner_loop_only");
    run_bench<true>(corpus, "bootstrap_path");

    return 0;
}
