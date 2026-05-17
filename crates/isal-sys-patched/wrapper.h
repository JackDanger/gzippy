#include "isa-l/include/igzip_lib.h"
#include "isa-l/igzip/huff_codes.h"

/* Declarations for ISA-L internals exposed by our patch
 * (igzip_inflate.c) — used by the parallel single-member bootstrap to
 * call ISA-L's fast Huffman table builder. Mirrors rapidgzip's
 * HuffmanCodingISAL wrapper (vendor/rapidgzip/.../huffman/HuffmanCodingISAL.hpp).
 */
int set_and_expand_lit_len_huffcode(struct huff_code *lit_len_huff,
                                    uint32_t table_length, uint16_t *count,
                                    uint16_t *expand_count, uint32_t *code_list);

void make_inflate_huff_code_lit_len(struct inflate_huff_code_large *result,
                                    struct huff_code *huff_code_table,
                                    uint32_t table_length,
                                    uint16_t *count_total,
                                    uint32_t *code_list,
                                    uint32_t multisym);

int set_codes(struct huff_code *huff_code_table, int table_length, uint16_t *count);

void make_inflate_huff_code_dist(struct inflate_huff_code_small *result,
                                 struct huff_code *huff_code_table,
                                 uint32_t table_length,
                                 const uint16_t *count_total,
                                 uint32_t max_symbol);
