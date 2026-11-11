# Record: post-Wave-B byte census — 2026-11-11

Branch `records/wave-B-byte-census` off `origin/main` @ `13c32862` (PRs #457
near-opt DP lever + #458 L9-12 -p2 route flip, both merged). Baseline computed
against a BASE worktree at `7e0d08cd` (the pre-Wave-B main, PR #456 merge).

**Scope: authoritative byte census of what ships.** Every number below is a
measured sha256 + byte size, produced by the commands in this file. No claim
about what the binary "should" do appears without the command that produced it
(charter hard stop #8).

## Setup

- MAIN binary:  `.review-tmp/wC-bytes/tgt-merged/release/gzippy`
  (built from `13c32862`, release, `--target-dir tgt-merged`;
  binary sha256 `f8b5f054ebdc35e4544ab1aeacf183d8aff39fad91b08468a418b1ebff8624cf`)
- BASE binary:  `.review-tmp/wC-bytes/tgt-base/release/gzippy`
  (built from `7e0d08cd`, release, `--target-dir tgt-base`;
  binary sha256 `f5afff2f2c91ac5c8ff8f3df69689545f26c42cd42dbeccd18a1db4bf18527d9`)
- Fresh `git worktree`s do not materialize submodules: linked via
  `scripts/dev/worktree-link.sh` + explicit `vendor/isa-l` symlink (the empty-dir
  case the script skips).
- Decoders: `/usr/bin/gzip` (Apple gzip 479, `-dc | shasum -a 256`),
  `vendor/pigz/pigz` 2.8 (`pigz -dc | shasum -a 256`), python3 zlib
  member-chain (`decode_sha.py`, sha-compares the concatenated decompressed
  bytes). `/opt/homebrew/bin/gzip --version` says `gzippy 0.1.7` — an installed
  argv0 alias, DEMOTED per the ab.sh warning; not used as a decoder.
- All compression runs stream mode (`-c`), so no FNAME/mtime in the header.
  Each stream verified by THREE decoders byte-roundtrip (sha-equal to the
  original input) — zero decode failures tolerated.
- Corpus: 30 members
  = 12 `benchmark_data/silesia.tar` members
  + `benchmark_data/logs.txt` (221,249,619 B)
  + `benchmark_data/software.archive` (221,249,546 B)
  + 8 even slices of each big file (27,656,203 B ×7 + tail; reconstruction
  sha-verified `cat logs.part-a? | shasum` == whole-file sha, same for sw).

## A. Anchors

Anchor stream command (stdin→stdout): `gzippy (-9|--level N) -pP -c < in16m.bin`
where `in16m.bin` = 16,085,866 B, sha256 `1010fa2f787d0a928fd10823085c62a9ee080fcff96b2608c5d216b2b8a1463c`
(`cat corpus/silesia_blocks/*.bin ×2 + test_data/text-1MB.txt ×2` — the pinned
ab.sh recipe; stamp verify pass, sha matches the 2026-11-10 probe record).

| # | cell | binary | expected | measured size | measured sha256 | verdict |
|---|---|---|---|---|---|---|
| A1 | `-9 -p2` | merged (13c32862) | 3,816,930 (wave-9 flip receipt) | 3,816,930 | `edc698cc16c8b91f50e59b17fd06a66b7e141bdee455be83017b3d9213b99e5a` | PASS |
| A1 | `-9 -p4` | merged (13c32862) | 3,740,910 / `327b083588efca21…` | 3,740,910 | `327b083588efca2178d97f977cbdca37e9a3815b7440aea5ccbe099e566c5475` | PASS |
| A2 | `-9 -p4` | base (7e0d08cd) | EXACTLY the merged sha (Lever A byte-exact) | 3,740,910 | `327b083588efca2178d97f977cbdca37e9a3815b7440aea5ccbe099e566c5475` | PASS (identical) |
| A2 | `-9 -p2` | base (7e0d08cd) | EXPECTED DIFFERENT (the #458 flip) | 3,740,910 | `327b083588efca2178d97f977cbdca37e9a3815b7440aea5ccbe099e566c5475` | RECORD-the-flip: base arm = pre-flip near-opt route (byte-identical to its -p4); merged arm = the flipped Lazy2-600 chunk arm, 3,816,930 B / `edc698cc…` — the flip shipped exactly as received |
| A3 | `-8 -p1` | merged (13c32862) | expected 3,821,170 (#453-era receipts) | 3,821,170 | `2a0fb084316ae7f6e5dd8034c729672a3ab520a57fe344a48f6e996e9c35025d` | PASS |
| A3 | `-8 -p4` | merged (13c32862) | expected 3,819,910 (#453-era receipts) | 3,819,910 | `b7090a22573814e2520aec366da9d89b5c7ec713f8034c026373cc016020299d` | PASS |
| A4 | `-6 -p2` | merged vs base | ports route unchanged — hashes match base | both | both `7997e3ace401da72542d80f41fdf4e6f0ea19e7ca9cf1fc0098ec8cdd7834190` | PASS |
| A4 | `-6 -p4` | merged vs base | ports route unchanged — hashes match base | both 3,848,412 | both `7997e3ace401da72542d80f41fdf4e6f0ea19e7ca9cf1fc0098ec8cdd7834190` | PASS |

All ten anchor streams roundtrip through the three independent decoders OK.

## C. Determinism

Four representative commands, each run twice (same cmd, fresh process):

| stream | cmd | run1 sha == run2 sha | verdict |
|---|---|---|---|
| det-a | `-9 -p2` | `edc698cc…` == `edc698cc…` (3,816,930 B) | DETERMINISTIC |
| det-b | `-9 -p4` | `327b0835…` == `327b0835…` (3,740,910 B) | DETERMINISTIC |
| det-c | `-8 -p3` | `b7090a22…` == `b7090a22…` (3,819,910 B) | DETERMINISTIC |
| det-d | `--level 12 -p4` | `bcabfc5fcb3c564111598905345a5e0d4ca888918bf8b2f324a4d953f2733037` == same | DETERMINISTIC |

(Side note: `-8 -p3` is byte-identical to `-8 -p4` — consistent with the
T2==T4 band at L8; see B.)

## B. The bands

Matrix: 30 members × L1..L8 × {T1,T2,T4} = 720 cells (section `b1`), then ×
L9..L12 × {T1,T2,T3,T4} = 480 cells (section `b2`) — 1,200 streams total on the
merged binary (13c32862). Every cell: compress `-c` → sha256, then decode
byte-roundtrip through all three decoders (sha-equality against the original
input). Each cell ran on its own fresh process; streams discarded after verify.

**Band verdicts**

| assertion | expected cells | asserted | result |
|---|---:|---:|---|
| L1..L8: `T2` stream == `T4` stream (byte-identical) | 240 | 240/240 | **PASS — the renegotiated band held, zero exceptions** |
| L9..L12: `T3` stream == `T4` stream (byte-identical) | 120 | 120/120 | **PASS — zero exceptions** |
| every stream byte-roundtrips through 3 independent decoders | 1,200 × 3 | 3,600/3,600 | **PASS — zero decode failures** |

**Verbatim FAIL rows: none.** (No failed expectation anywhere in A/B/C —
there was no D-rule capture to make.)

Decode matrix (per decoder, all 1,200 band streams + 10 anchor streams):

| decoder | role | tested | byte-sha matched | failures |
|---|---|---:|---:|---:|
| `/usr/bin/gzip -dc` (Apple gzip 479; the vendored GNU gzip aborts at `--version` on this box and was NOT used) | vendor gzip | 1,210 | 1,210 | 0 |
| `vendor/pigz/pigz 2.8 -dc` | vendor pigz | 1,210 | 1,210 | 0 |
| python3 zlib member-chain (decode_sha.py) | independent library | 1,210 | 1,210 | 0 |

Counts: 10 anchor streams + 1,200 band streams = 1,210 per decoder leg.

**Per-member size table** (sizes at `-c` stdout; T4 column for L1-L8 is the
same bytes as T2 — asserted; T2 deep columns are the Lazy2-600 chunk arm):

| member | orig B | L1 T4 | L2 T4 | L3 T4 | L4 T4 | L5 T4 | L6 T4 | L7 T4 | L8 T4 | L9 T4 | L10 T4 | L11 T4 | L12 T4 | L9 T2 | L10 T2 | L11 T2 | L12 T2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `silesia-dickens` | 10,192,446 | 4128333 | 3959153 | 3864090 | 3850283 | 3837971 | 3829670 | 3829550 | 3812146 | 3687186 | 3687233 | 3683397 | 3683018 | 3810740 | 3810740 | 3810740 | 3810740 |
| `silesia-mozilla` | 51,220,480 | 19512602 | 19202435 | 18875722 | 18782392 | 18755282 | 18701123 | 18693213 | 18686467 | 18246993 | 18252140 | 18227954 | 18226824 | 18698390 | 18698390 | 18698390 | 18698390 |
| `silesia-mr` | 9,970,564 | 3754435 | 3643345 | 3649171 | 3643009 | 3635879 | 3632428 | 3632427 | 3624917 | 3462798 | 3462912 | 3448328 | 3445901 | 3624554 | 3624554 | 3624554 | 3624554 |
| `silesia-nci` | 33,553,445 | 3680514 | 3668083 | 3412091 | 3200360 | 3183455 | 2996542 | 2963422 | 2949364 | 2772212 | 2779109 | 2753859 | 2751348 | 2925538 | 2925538 | 2925538 | 2925538 |
| `silesia-ooffice` | 6,152,192 | 3140345 | 3121725 | 3073774 | 3070734 | 3068622 | 3065695 | 3065265 | 3062170 | 2995725 | 2995910 | 2994269 | 2993997 | 3066853 | 3066853 | 3066853 | 3066853 |
| `silesia-osdb` | 10,085,684 | 3753540 | 3806345 | 3685649 | 3660904 | 3660671 | 3659688 | 3659529 | 3652398 | 3610692 | 3611214 | 3609102 | 3609060 | 3654860 | 3654860 | 3654860 | 3654860 |
| `silesia-reymont` | 6,627,202 | 2112580 | 1996831 | 1898332 | 1863626 | 1836782 | 1811725 | 1809733 | 1809844 | 1706261 | 1706469 | 1702489 | 1701533 | 1806194 | 1806194 | 1806194 | 1806194 |
| `silesia-samba` | 21,606,400 | 5696347 | 5612057 | 5454050 | 5377476 | 5368982 | 5312148 | 5299270 | 5270242 | 5130020 | 5136568 | 5124842 | 5122759 | 5268074 | 5268074 | 5268074 | 5268074 |
| `silesia-sao` | 7,251,944 | 5421431 | 5374998 | 5333567 | 5330521 | 5325595 | 5321545 | 5321336 | 5320143 | 5251721 | 5251721 | 5250923 | 5250912 | 5322494 | 5322494 | 5322494 | 5322494 |
| `silesia-webster` | 41,458,703 | 13193871 | 12892434 | 12428607 | 12144104 | 12083947 | 11972814 | 11969711 | 11896084 | 11549526 | 11549505 | 11530955 | 11528806 | 11884039 | 11884039 | 11884039 | 11884039 |
| `silesia-x-ray` | 8,474,240 | 5985027 | 5993099 | 5981133 | 5998105 | 5998104 | 5998099 | 5998099 | 5962541 | 5763957 | 5763957 | 5753923 | 5753486 | 5986946 | 5986946 | 5986946 | 5986946 |
| `silesia-xml` | 5,345,280 | 847881 | 813841 | 752880 | 697569 | 690286 | 661541 | 655504 | 651176 | 634593 | 637995 | 631167 | 629584 | 649724 | 649724 | 649724 | 649724 |
| `logs.txt` | 221,249,619 | 11397783 | 12697489 | 12098995 | 12094534 | 12094946 | 10732998 | 8628908 | 8596823 | 8462704 | 9735007 | 8456455 | 8450419 | 8596828 | 8596828 | 8596828 | 8596828 |
| `software.archive` | 221,249,546 | 6119062 | 5634585 | 5178337 | 5125579 | 5125487 | 5125936 | 4998796 | 4404956 | 5102984 | 5102984 | 4437264 | 4299945 | 4404956 | 4404956 | 4404956 | 4404956 |
| `logs.part-aa` | 27,656,203 | 1427751 | 1593265 | 1517910 | 1516311 | 1516363 | 1344441 | 1081113 | 1077010 | 1060072 | 1219431 | 1058999 | 1058935 | 1077012 | 1077012 | 1077012 | 1077012 |
| `logs.part-ab` | 27,656,203 | 1425163 | 1587051 | 1512538 | 1512022 | 1512073 | 1341786 | 1078872 | 1074838 | 1057954 | 1217135 | 1056699 | 1056595 | 1074839 | 1074839 | 1074839 | 1074839 |
| `logs.part-ac` | 27,656,203 | 1425116 | 1587231 | 1512686 | 1512184 | 1512236 | 1341931 | 1079019 | 1074919 | 1057735 | 1217143 | 1056654 | 1056575 | 1074920 | 1074920 | 1074920 | 1074920 |
| `logs.part-ad` | 27,656,203 | 1425489 | 1587148 | 1512390 | 1511966 | 1512016 | 1341941 | 1078992 | 1074937 | 1057788 | 1217217 | 1056640 | 1056709 | 1074938 | 1074938 | 1074938 | 1074938 |
| `logs.part-ae` | 27,656,203 | 1424481 | 1587063 | 1512338 | 1511935 | 1511988 | 1341882 | 1079062 | 1074981 | 1057674 | 1217345 | 1056578 | 1056714 | 1074981 | 1074981 | 1074981 | 1074981 |
| `logs.part-af` | 27,656,203 | 1425758 | 1587263 | 1512759 | 1512313 | 1512366 | 1341997 | 1079115 | 1075018 | 1057783 | 1217310 | 1056785 | 1056606 | 1075019 | 1075019 | 1075019 | 1075019 |
| `logs.part-ag` | 27,656,203 | 1425216 | 1587306 | 1512617 | 1512128 | 1512178 | 1341902 | 1079091 | 1074995 | 1057841 | 1217213 | 1056833 | 1056572 | 1074997 | 1074997 | 1074997 | 1074997 |
| `logs.part-ah` | 27,656,198 | 1422349 | 1583109 | 1508513 | 1508253 | 1508306 | 1339093 | 1076736 | 1072722 | 1055549 | 1214510 | 1054440 | 1054434 | 1072724 | 1072724 | 1072724 | 1072724 |
| `sw.part-aa` | 27,656,203 | 765533 | 704941 | 647937 | 641303 | 641295 | 641339 | 625458 | 551236 | 638365 | 638365 | 555177 | 538039 | 551236 | 551236 | 551236 | 551236 |
| `sw.part-ab` | 27,656,203 | 765597 | 704958 | 648019 | 641341 | 641322 | 641351 | 625454 | 551247 | 638392 | 638392 | 555255 | 538047 | 551247 | 551247 | 551247 | 551247 |
| `sw.part-ac` | 27,656,203 | 765539 | 704937 | 647925 | 641333 | 641317 | 641340 | 625439 | 551240 | 638362 | 638362 | 555226 | 537998 | 551240 | 551240 | 551240 | 551240 |
| `sw.part-ad` | 27,656,203 | 765576 | 704952 | 647953 | 641314 | 641302 | 641341 | 625453 | 551241 | 638375 | 638375 | 555224 | 538035 | 551241 | 551241 | 551241 | 551241 |
| `sw.part-ae` | 27,656,203 | 765576 | 704946 | 647955 | 641309 | 641297 | 641320 | 625441 | 551244 | 638363 | 638363 | 555174 | 538029 | 551244 | 551244 | 551244 | 551244 |
| `sw.part-af` | 27,656,203 | 765615 | 704952 | 647946 | 641370 | 641347 | 641378 | 625469 | 551248 | 638373 | 638373 | 555211 | 538022 | 551248 | 551248 | 551248 | 551248 |
| `sw.part-ag` | 27,656,203 | 765609 | 704946 | 647982 | 641337 | 641331 | 641355 | 625460 | 551246 | 638374 | 638374 | 555240 | 538050 | 551246 | 551246 | 551246 | 551246 |
| `sw.part-ah` | 27,656,125 | 765571 | 704949 | 647932 | 641307 | 641297 | 641334 | 625450 | 551237 | 638360 | 638360 | 555201 | 538042 | 551237 | 551237 | 551237 | 551237 |

Total streams: 1200 OK, 0 FAIL



## Files

- `sizes.tsv` — all 1,214 recorded streams with sha256 + size (10 anchors,
  1,200 band cells, 4 determinism rows).
- In-repo unfixed data (not committed here): `.review-tmp/wC-bytes/run/`
  (`anchors.tsv`, `band.tsv`, `det.tsv`, `parity.txt`, `band_stream_sizes.tsv`,
  `members.tsv`) — raw dumps backing every number above.

## Analysis notes (census observations beyond the assertions)

- Deep-level structure, measured on the 30-member corpus: the T2 streams are
  byte-identical ACROSS L9..L12 for every member (30/30; e.g. `logs.txt` T2 =
  8,596,828 B with sha identical at L9, L10, L11 and L12) — consistent with the
  #458 flip routing threads<=2 at those levels to the same `params_inner(9)`
  Lazy2-600 chunk arm. At L9..L12, T1 is byte-different from T2 on all 30
  members (0/120 pairs): T1 is the per-level whole-stream route (L9's T1
  byte-equals L8's T1, sha `4d8a921c…` on software.archive; L10/L12 T1 run
  near-opt, L11 zopfli), while T2 is the chunked Lazy2 arm.
- `-8 -p3` is byte-identical to `-8 -p4` on in16m (determinism det-c) —
  consistent with the T2==T4 band asserted 240/240 at L1-L8.
- L1-beats-L2 rows exist and are RECORDED, not adjudicated: several members
  compress smaller at L1 than L2 (e.g. `silesia-osdb` 3,753,540 → 3,806,345;
  `silesia-x-ray` 5,985,027 → 5,993,099; `logs.txt` 11,397,783 → 12,697,489).
  Census scope: no assertion above depends on monotonicity.

