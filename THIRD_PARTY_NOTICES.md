# Third-party notices

gzippy itself is released under the [zlib license](LICENSE), copyright
Jack Danger. Substantial portions of the source are **ports or
transliterations** of code from the projects below, and their license terms
apply to the corresponding files. Every ported file carries a module-level
doc comment naming the exact upstream file (and usually line range) it was
ported from; this document collects the license and copyright notices those
upstreams require.

"Port" here means a function-by-function (sometimes line-by-line) translation
into Rust — a derivative work of the upstream source. Files listed as
"inspiration / ideas only" contain no translated upstream code; they are
listed as a courtesy and to be conservative.

| Upstream | License | Copyright | Derived code in this repo |
|---|---|---|---|
| [libdeflate](https://github.com/ebiggers/libdeflate) | MIT | 2016 Eric Biggers, 2024 Google LLC | `src/compress/ldx/` (entire tree), parts of `src/compress/deflate/`, two decode-side files |
| [Zopfli](https://github.com/google/zopfli) | Apache-2.0 | 2011 Google Inc. | `src/compress/deflate/parse/ultra/` (most of the tree), `src/compress/deflate/huffman/optimal.rs` |
| [rapidgzip](https://github.com/mxmlnkn/rapidgzip) | MIT (dual MIT/Apache-2.0; MIT elected) | 2019-2023 Maximilian Knespel | `src/decompress/parallel/` (most of the tree) |
| [ISA-L / igzip](https://github.com/intel/isa-l) | BSD-3-Clause | 2011-2024 Intel Corporation | matchfinder in `src/compress/deflate/parse/fast.rs`; `src/decompress/parallel/lut_huffman.rs` |
| [zlib](https://zlib.net) / [zlib-ng](https://github.com/zlib-ng/zlib-ng) | zlib | 1995-2024 Jean-loup Gailly and Mark Adler (and zlib-ng contributors) | tuning heuristics/constants in `src/compress/deflate/level.rs`, `parse/greedy.rs`, `parse/lazy.rs`, `matchfinder/hc.rs` |
| [Efficient Compression Tool (ECT)](https://github.com/fhanau/Efficient-Compression-Tool) | Apache-2.0 | Felix Hanau | `src/compress/deflate/matchfinder/lzfind.rs` |
| LZMA SDK `LzFind.c` (via ECT) | Public domain | Igor Pavlov (2009) | `src/compress/deflate/matchfinder/lzfind.rs` |
| [pigz](https://zlib.net/pigz/) | zlib-style | 2007-2023 Mark Adler | inspiration / ideas only (parallel-compression architecture, CLI behaviour); no ported code identified |

---

## libdeflate — MIT

Upstream: <https://github.com/ebiggers/libdeflate> (vendored at
`vendor/libdeflate`). License: MIT.

Ported / derived files:

- `src/compress/ldx/` — **the entire module** is a declared faithful,
  function-by-function port of `lib/deflate_compress.c` (see
  `src/compress/ldx/mod.rs`); each item carries a `C:` comment naming its
  source line.
- `src/compress/ldx_oracle.rs` — differential-test harness over the port.
- `src/compress/deflate/bitstream.rs` — output bitstream machinery.
- `src/compress/deflate/block_split.rs` — `block_split_stats` heuristic.
- `src/compress/deflate/costs.rs` — near-optimal cost model.
- `src/compress/deflate/tables.rs` — DEFLATE constant tables.
- `src/compress/deflate/level.rs` — level preset table (also carries zlib-ng
  derived knobs, see below).
- `src/compress/deflate/huffman/fast.rs` — approximate length-limited
  Huffman builder.
- `src/compress/deflate/huffman/header.rs` — dynamic-block header (precode)
  construction.
- `src/compress/deflate/matchfinder/common.rs` — shared matchfinding
  primitives (`lz_hash`, `lz_extend`, init/rebase).
- `src/compress/deflate/matchfinder/hc.rs` — hash-chains matchfinder.
- `src/compress/deflate/matchfinder/ht.rs` — hash-table matchfinder
  (extended with a `hash3_tab` in the shape of libdeflate's own
  `hc_matchfinder`).
- `src/compress/deflate/matchfinder/bt.rs` — binary-tree matchfinder.
- `src/compress/deflate/parse/greedy.rs`, `parse/lazy.rs`,
  `parse/ht_fast.rs`, `parse/near_optimal.rs` — the greedy, lazy/lazy2,
  fastest, and near-optimal parsers.
- `src/decompress/inflate/libdeflate_entry.rs` — reimplements libdeflate's
  Huffman table-entry format (`lib/deflate_decompress.c`).
- `src/decompress/inflate/consume_first_decode.rs` — written to match the
  structure of libdeflate's `lib/decompress_template.h`.

```
Copyright 2016 Eric Biggers
Copyright 2024 Google LLC

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Zopfli — Apache-2.0

Upstream: <https://github.com/google/zopfli> (vendored at `vendor/zopfli`).
License: Apache License, Version 2.0 (full text in the appendix below).
Copyright 2011 Google Inc. All Rights Reserved.

Ported / derived files (the `-10`/`-11`/`-12` "crown" engine):

- `src/compress/deflate/parse/ultra/blocksplit.rs` — `blocksplitter.c`
  (its cost callback follows ECT's variant; see the file header).
- `src/compress/deflate/parse/ultra/cache.rs` — `cache.c`.
- `src/compress/deflate/parse/ultra/deflate.rs` — bit-emitting half of
  `deflate.c`.
- `src/compress/deflate/parse/ultra/deflate_size.rs` — size-estimation half
  of `deflate.c`.
- `src/compress/deflate/parse/ultra/gzip.rs` — `gzip_container.c`.
- `src/compress/deflate/parse/ultra/hash.rs` — `hash.c`.
- `src/compress/deflate/parse/ultra/lz77.rs` — `lz77.c`.
- `src/compress/deflate/parse/ultra/squeeze.rs` — `squeeze.c`.
- `src/compress/deflate/parse/ultra/symbols.rs` — `symbols.h`.
- `src/compress/deflate/parse/ultra/zlib.rs` — `zlib_container.c`.
- `src/compress/deflate/huffman/optimal.rs` — `katajainen.c` (bounded
  package-merge) and the non-entropy half of `tree.c`.

Statement of changes (Apache-2.0 §4(b)): these files are Rust translations
of the C sources named above, restructured into gzippy's module layout,
extended with multi-threaded block splitting, LzFind/BT4 matchfinding,
multi-seed iterated squeeze, and gzippy-specific tuning; they are not the
original Zopfli source. Upstream Zopfli distributes no NOTICE file.

## rapidgzip — MIT

Upstream: <https://github.com/mxmlnkn/rapidgzip> (vendored at
`vendor/rapidgzip`). Dual-licensed MIT OR Apache-2.0; gzippy uses it under
**MIT**. Copyright (c) 2019-2023 Maximilian Knespel. The MIT terms are the
same as printed for libdeflate above, with this copyright line.

Ported / derived files — the parallel single-member decode tree,
`src/decompress/parallel/`. `mod.rs` there holds the full gzippy-module →
rapidgzip-source role map; the files declaring literal or structural ports
include: `apply_window.rs`, `async_block_finder.rs`, `bit_manipulation.rs`,
`bit_reader.rs`, `block_fetcher.rs`, `block_map.rs`,
`blockfinder_validation.rs`, `cache.rs`, `chunk_data.rs`, `chunk_decode.rs`,
`chunk_fetcher.rs`, `compressed_vector.rs`, `crc32.rs`, `error.rs`,
`gzip_block_finder.rs`, `gzip_definitions.rs`, `gzip_format.rs`,
`huffman_base.rs`, `huffman_reversed_bits_cached.rs`,
`huffman_short_bits_cached.rs`, `huffman_symbols_per_length.rs`,
`inflate_wrapper.rs`, `marker_inflate.rs`, `prefetcher.rs`,
`replace_markers.rs`, `segmented_buffer.rs`, `segmented_markers.rs`,
`statistics.rs`, `streamed_results.rs`, `thread_pool.rs`,
`used_window_symbols.rs`, `width_ring.rs`, `window_map.rs`.

## Intel ISA-L (igzip) — BSD-3-Clause

Upstream: <https://github.com/intel/isa-l> (vendored at `vendor/isa-l`).
License: BSD-3-Clause. Copyright(c) 2011-2024 Intel Corporation.

Ported / derived files:

- `src/compress/deflate/parse/fast.rs` — the L0/L1 chainless single-probe
  matchfinder is a declared port of `igzip/igzip_base.c`
  (`isal_deflate_body_base`) with named deviations; block emission is
  gzippy's own shared backend.
- `src/decompress/parallel/lut_huffman.rs` — pure-Rust port of ISA-L's
  `inflate_huff_code_{large,small}` LUT format and its builder functions.

```
Copyright(c) 2011-2024 Intel Corporation All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
  * Neither the name of Intel Corporation nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## zlib and zlib-ng — zlib license

Upstreams: <https://zlib.net> and <https://github.com/zlib-ng/zlib-ng>
(vendored at `vendor/zlib-ng`). License: zlib (the same license gzippy uses;
see `LICENSE` for the text). (C) 1995-2024 Jean-loup Gailly and Mark Adler,
and zlib-ng contributors.

Derived material — search-strategy heuristics and tuning constants, not
literal code translation:

- `src/compress/deflate/level.rs` — zlib-ng `configuration_table` chain
  depths and `good_length` ("good_match") values for the T1 L5-L7 path.
- `src/compress/deflate/parse/lazy.rs`, `parse/greedy.rs`,
  `matchfinder/hc.rs` — the `good_match` chain-quartering rule
  (`match_tpl.h`) and `TOO_FAR` far-offset guard precedents.

Per zlib license terms: this is altered material, plainly marked as such,
and is not the original zlib/zlib-ng software.

## Efficient Compression Tool (ECT) and LZMA SDK — Apache-2.0 / public domain

Upstream: <https://github.com/fhanau/Efficient-Compression-Tool> (Felix
Hanau), Apache License 2.0. ECT's `src/LzFind.c` is itself derived from the
LZMA SDK: "LzFind.c -- Match finder for LZ algorithms / 2009-04-22 : Igor
Pavlov : Public domain", with modifications by Felix Hanau.

Ported / derived files:

- `src/compress/deflate/matchfinder/lzfind.rs` — faithful Rust port of
  ECT's trimmed `LzFind.c` (`Bt3Zip` binary-tree matchfinder), with
  deviations documented in the file header.
- `src/compress/deflate/parse/ultra/blocksplit.rs` — adopts ECT's
  blocksplitter cost-callback choice (idea-level; the code is the Zopfli
  port listed above).

Statement of changes (Apache-2.0 §4(b)): `lzfind.rs` is a Rust translation
of the C source, with safe-bounds handling and without ECT's
`mfinexport` cross-block carry; it is not the original ECT source.

## pigz — inspiration only

<https://zlib.net/pigz/>, by Mark Adler (zlib-style license, (C) 2007-2023).
gzippy's parallel-compression architecture (independent compression workers
feeding a single writer) and CLI behaviour follow pigz's design, and the
`--rsyncable` and `-b` block-size semantics match pigz's documented
behaviour. No pigz code has been identified as ported; this credit is
acknowledgment of design influence.

---

## Appendix: Apache License, Version 2.0

Applies to the Zopfli- and ECT-derived files listed above.

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
```
