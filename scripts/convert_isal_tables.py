#!/usr/bin/env python3
"""
Convert ISA-L C lookup tables to Rust format.

This script:
1. Parses C header files containing static lookup tables
2. Generates multi-symbol Huffman decode entries (ISA-L's key optimization)
3. Outputs Rust const arrays

Multi-symbol format (32-bit entry):
  For single/long codes (flag bit = 1):
    Bits 0-9:   Symbol
    Bit  25:    Flag (1 = needs long lookup or single symbol)
    Bits 28-31: Code length

  For multi-symbol (flag bit = 0):
    Bits 0-7:   First literal
    Bits 8-15:  Second literal  
    Bits 16-23: Third literal (if count = 3)
    Bit  25:    Flag (0 = multi-symbol)
    Bits 26-27: Symbol count - 1 (0=1, 1=2, 2=3)
    Bits 28-31: Total code length
"""

import re
import sys
from pathlib import Path


# Constants matching ISA-L
ISAL_DECODE_LONG_BITS = 12
LARGE_FLAG_BIT = 1 << 25
LARGE_SYM_COUNT_OFFSET = 26
LARGE_SHORT_CODE_LEN_OFFSET = 28


def extract_hex_array(text: str) -> list[int]:
    """Extract hex values from a C array initializer."""
    hex_pattern = r'0x[0-9a-fA-F]+'
    matches = re.findall(hex_pattern, text)
    return [int(m, 16) for m in matches]


def parse_struct_array(content: str, struct_name: str, field_name: str) -> list[int] | None:
    """Parse a field from a C struct initializer."""
    struct_pattern = rf'static\s+struct\s+\w+\s+{struct_name}\s*=\s*\{{([^;]+)\}};'
    struct_match = re.search(struct_pattern, content, re.DOTALL)
    if not struct_match:
        return None
    
    struct_body = struct_match.group(1)
    field_pattern = rf'\.{field_name}\s*=\s*\{{\s*([^}}]+)\}}'
    field_match = re.search(field_pattern, struct_body, re.DOTALL)
    if not field_match:
        return None
    
    return extract_hex_array(field_match.group(1))


def infer_rust_type(values: list[int]) -> str:
    """Infer the appropriate Rust type for the values."""
    max_val = max(values) if values else 0
    if max_val <= 0xFF:
        return "u8"
    elif max_val <= 0xFFFF:
        return "u16"
    else:
        return "u32"


def to_rust_array(name: str, values: list[int], rust_type: str = None) -> str:
    """Convert values to a Rust const array."""
    if rust_type is None:
        rust_type = infer_rust_type(values)
    
    rust_name = name.upper()
    
    lines = []
    for i in range(0, len(values), 8):
        chunk = values[i:i+8]
        if rust_type == "u8":
            line = ", ".join(f"0x{v:02x}" for v in chunk)
        elif rust_type == "u16":
            line = ", ".join(f"0x{v:04x}" for v in chunk)
        else:
            line = ", ".join(f"0x{v:08x}" for v in chunk)
        lines.append(f"    {line},")
    
    values_str = "\n".join(lines)
    return f"pub const {rust_name}: [{rust_type}; {len(values)}] = [\n{values_str}\n];"


# =============================================================================
# Fixed Huffman Code Tables (RFC 1951)
# =============================================================================

def build_fixed_lit_len_codes():
    """Build fixed Huffman codes for literal/length symbols (RFC 1951 section 3.2.6)."""
    codes = []
    
    # 0-143: 8-bit codes starting at 00110000
    for i in range(144):
        code = 0b00110000 + i
        codes.append((i, code, 8))
    
    # 144-255: 9-bit codes starting at 110010000
    for i in range(144, 256):
        code = 0b110010000 + (i - 144)
        codes.append((i, code, 9))
    
    # 256-279: 7-bit codes starting at 0000000
    for i in range(256, 280):
        code = 0b0000000 + (i - 256)
        codes.append((i, code, 7))
    
    # 280-287: 8-bit codes starting at 11000000
    for i in range(280, 288):
        code = 0b11000000 + (i - 280)
        codes.append((i, code, 8))
    
    return codes


def reverse_bits(val: int, n: int) -> int:
    """Reverse n bits in val."""
    result = 0
    for _ in range(n):
        result = (result << 1) | (val & 1)
        val >>= 1
    return result


def build_multi_symbol_table():
    """
    Build a multi-symbol lookup table for fixed Huffman codes.
    
    This is ISA-L's key optimization: pack 2-3 literals into a single lookup.
    """
    codes = build_fixed_lit_len_codes()
    
    # Build reverse lookup: reversed_code -> (symbol, length)
    code_map = {}
    for sym, code, length in codes:
        rev = reverse_bits(code, length)
        code_map[(rev, length)] = sym
    
    # Build 12-bit lookup table
    table_size = 1 << ISAL_DECODE_LONG_BITS
    table = [0] * table_size
    
    for lookup_bits in range(table_size):
        entry = build_entry_for_bits(lookup_bits, code_map, codes)
        table[lookup_bits] = entry
    
    return table


def build_entry_for_bits(lookup_bits: int, code_map: dict, codes: list) -> int:
    """Build a table entry for the given 12-bit lookup value."""
    
    # Try to decode first symbol
    sym1, len1 = decode_symbol(lookup_bits, code_map)
    
    if sym1 is None:
        # Invalid code
        return LARGE_FLAG_BIT  # Flag bit set, zero length = invalid
    
    if sym1 >= 256:
        # Length code or end of block - can't pack more symbols
        return pack_single_symbol(sym1, len1)
    
    # First symbol is a literal - try to pack more
    remaining_bits = lookup_bits >> len1
    remaining_len = ISAL_DECODE_LONG_BITS - len1
    
    if remaining_len <= 0:
        return pack_single_symbol(sym1, len1)
    
    # Try second symbol
    sym2, len2 = decode_symbol_partial(remaining_bits, remaining_len, code_map)
    
    if sym2 is None or sym2 >= 256:
        # Can't decode second or it's not a literal
        return pack_single_symbol(sym1, len1)
    
    total_len = len1 + len2
    if total_len > ISAL_DECODE_LONG_BITS:
        return pack_single_symbol(sym1, len1)
    
    # Try third symbol
    remaining_bits2 = remaining_bits >> len2
    remaining_len2 = remaining_len - len2
    
    if remaining_len2 <= 0:
        return pack_double_symbol(sym1, sym2, total_len)
    
    sym3, len3 = decode_symbol_partial(remaining_bits2, remaining_len2, code_map)
    
    if sym3 is None or sym3 >= 256:
        return pack_double_symbol(sym1, sym2, total_len)
    
    total_len3 = total_len + len3
    if total_len3 > ISAL_DECODE_LONG_BITS:
        return pack_double_symbol(sym1, sym2, total_len)
    
    return pack_triple_symbol(sym1, sym2, sym3, total_len3)


def decode_symbol(bits: int, code_map: dict) -> tuple:
    """Decode a symbol from bits, trying shortest match first."""
    # Try lengths in order - we want the shortest valid match
    # Fixed Huffman: 7-bit (256-279), 8-bit (0-143, 280-287), 9-bit (144-255)
    for length in [7, 8, 9]:
        masked = bits & ((1 << length) - 1)
        if (masked, length) in code_map:
            return (code_map[(masked, length)], length)
    return (None, 0)


def decode_symbol_partial(bits: int, max_len: int, code_map: dict) -> tuple:
    """Decode a symbol from bits with limited available bits."""
    for length in range(7, min(10, max_len + 1)):
        if length > max_len:
            break
        masked = bits & ((1 << length) - 1)
        if (masked, length) in code_map:
            return (code_map[(masked, length)], length)
    return (None, 0)


def pack_single_symbol(sym: int, length: int) -> int:
    """Pack a single symbol into table entry."""
    # Set flag bit to indicate single symbol (or length code)
    # Bits 0-9: symbol, Bits 28-31: length
    return (sym & 0x3FF) | LARGE_FLAG_BIT | (length << LARGE_SHORT_CODE_LEN_OFFSET)


def pack_double_symbol(sym1: int, sym2: int, total_len: int) -> int:
    """Pack two literals into table entry."""
    # Flag bit = 0 (multi-symbol)
    # Bits 0-7: sym1, Bits 8-15: sym2
    # Bits 26-27: count - 1 = 1
    # Bits 28-31: total length
    return (
        (sym1 & 0xFF) |
        ((sym2 & 0xFF) << 8) |
        (1 << LARGE_SYM_COUNT_OFFSET) |
        (total_len << LARGE_SHORT_CODE_LEN_OFFSET)
    )


def pack_triple_symbol(sym1: int, sym2: int, sym3: int, total_len: int) -> int:
    """Pack three literals into table entry."""
    # Flag bit = 0 (multi-symbol)
    # Bits 0-7: sym1, Bits 8-15: sym2, Bits 16-23: sym3
    # Bits 26-27: count - 1 = 2
    # Bits 28-31: total length
    return (
        (sym1 & 0xFF) |
        ((sym2 & 0xFF) << 8) |
        ((sym3 & 0xFF) << 16) |
        (2 << LARGE_SYM_COUNT_OFFSET) |
        (total_len << LARGE_SHORT_CODE_LEN_OFFSET)
    )


# =============================================================================
# Distance Code Table
# =============================================================================

def build_fixed_dist_table():
    """Build fixed distance code lookup table."""
    # Fixed distance codes are 5 bits, values 0-29
    table_size = 1 << 10  # 10-bit lookup for consistency
    table = [0] * table_size
    
    for i in range(table_size):
        # Distance codes are 5 bits, reversed
        dist_code = reverse_bits(i & 0x1F, 5)
        if dist_code < 30:
            # Pack: symbol in bits 0-4, length (5) in bits 11-14
            table[i] = dist_code | (5 << 11)
        else:
            table[i] = 0x8000  # Invalid
    
    return table


# =============================================================================
# Main Output
# =============================================================================

def convert_rfc1951_tables(isal_path: Path) -> str:
    """Convert the RFC 1951 tables from igzip_inflate.c."""
    inflate_file = isal_path / "igzip" / "igzip_inflate.c"
    content = inflate_file.read_text()
    
    output_lines = [
        "// RFC 1951 deflate tables.",
        "// These tables are from ISA-L's igzip_inflate.c.",
        "",
    ]
    
    for field_name, rust_name, rust_type, desc in [
        ("dist_extra_bit_count", "DIST_EXTRA_BITS", "u8", "Extra bits for distance codes"),
        ("dist_start", "DIST_START", "u32", "Base distance for each distance code"),
        ("len_extra_bit_count", "LEN_EXTRA_BITS", "u8", "Extra bits for length codes"),
        ("len_start", "LEN_START", "u16", "Base length for each length code"),
    ]:
        values = parse_struct_array(content, "rfc_lookup_table", field_name)
        if values:
            output_lines.append(f"/// {desc}.")
            output_lines.append(to_rust_array(rust_name, values, rust_type))
            output_lines.append("")
    
    return "\n".join(output_lines)


def convert_code_order_table() -> str:
    """Generate the code length order table from RFC 1951."""
    order = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
    
    output_lines = [
        "// Code length order for dynamic Huffman headers.",
        "// Per RFC 1951 section 3.2.7, code lengths are stored in this order.",
        "",
        "/// Order of code length codes in dynamic Huffman blocks.",
        to_rust_array("CODE_LENGTH_ORDER", order, "u8"),
        "",
    ]
    return "\n".join(output_lines)


def generate_multi_symbol_tables() -> str:
    """Generate multi-symbol lookup tables."""
    output_lines = [
        "// Multi-symbol Huffman lookup tables.",
        "// These enable decoding 2-3 literals in a single lookup (ISA-L's key optimization).",
        "//",
        "// Entry format (32-bit):",
        "//   Multi-symbol (flag bit 25 = 0):",
        "//     Bits 0-7:   First literal",
        "//     Bits 8-15:  Second literal",
        "//     Bits 16-23: Third literal (if count = 3)",
        "//     Bits 26-27: Symbol count - 1",
        "//     Bits 28-31: Total code length",
        "//   Single symbol (flag bit 25 = 1):",
        "//     Bits 0-9:   Symbol (0-285)",
        "//     Bits 28-31: Code length",
        "",
    ]
    
    # Generate multi-symbol table
    table = build_multi_symbol_table()
    output_lines.append("/// Multi-symbol fixed Huffman lookup table (4096 entries).")
    output_lines.append(to_rust_array("MULTI_SYM_LIT_TABLE", table, "u32"))
    output_lines.append("")
    
    # Generate distance table
    dist_table = build_fixed_dist_table()
    output_lines.append("/// Fixed distance code lookup table (1024 entries).")
    output_lines.append(to_rust_array("FIXED_DIST_TABLE", dist_table, "u16"))
    output_lines.append("")
    
    # Add constants
    output_lines.append("// Constants for multi-symbol decode.")
    output_lines.append(f"pub const LARGE_FLAG_BIT: u32 = 0x{LARGE_FLAG_BIT:08x};")
    output_lines.append(f"pub const LARGE_SYM_COUNT_OFFSET: u32 = {LARGE_SYM_COUNT_OFFSET};")
    output_lines.append(f"pub const LARGE_CODE_LEN_OFFSET: u32 = {LARGE_SHORT_CODE_LEN_OFFSET};")
    output_lines.append("")
    
    return "\n".join(output_lines)


def main():
    if len(sys.argv) > 1:
        isal_path = Path(sys.argv[1])
    else:
        script_dir = Path(__file__).parent
        isal_path = script_dir.parent / "isa-l"
    
    if not isal_path.exists():
        print(f"Error: ISA-L path not found: {isal_path}", file=sys.stderr)
        sys.exit(1)
    
    print("//! Auto-generated ISA-L lookup tables for Rust.")
    print("//!")
    print("//! Generated by scripts/convert_isal_tables.py")
    print("//! Do not edit manually!")
    print("")
    print("#![allow(dead_code)]")
    print("")
    
    # Convert RFC 1951 tables
    print(convert_rfc1951_tables(isal_path))
    
    # Generate code order table
    print(convert_code_order_table())
    
    # Generate multi-symbol tables
    print(generate_multi_symbol_tables())


if __name__ == "__main__":
    main()
