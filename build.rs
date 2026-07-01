//! Build script for gzippy.
//!
//! gzippy uses only statically-linked dependencies for safe distribution:
//! - libdeflate (via libdeflater crate) - statically linked, highly optimized
//! - zlib-ng (via flate2 with zlib-ng feature) - statically linked
//!
//! Zopfli was previously linked as a vendored C library; since the
//! pure-Rust port at `src/backends/zopfli_pure` reached parity (Steps
//! 1–23 of plan.md) the C build is gone for production.
//!
//! The `oracle` cargo feature opts back into compiling
//! `vendor/zopfli/src/zopfli/*.c` (gated to test builds only) so the
//! Phase 11.2 corpus oracle can compare zopfli_pure's output to the C
//! reference byte-for-byte. See `src/backends/zopfli_pure/oracle_tests.rs`.
//!
//! Note: ISA-L FFI was attempted but the complex struct layout (200KB+ with
//! nested Huffman tables) makes Rust FFI difficult. Since ISA-L uses pure C
//! fallback on ARM anyway, libdeflate is the pragmatic choice.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=scripts/pre-commit");
    println!("cargo:rerun-if-changed=scripts/pre-push");
    install_git_hooks();

    emit_parallel_sm_cfgs();

    if std::env::var("CARGO_FEATURE_ORACLE").is_ok() {
        build_zopfli_oracle();
    }
}

/// Emit the compile-time `cfg` aliases that gate the parallel single-member
/// decode pipeline. Centralizing the predicate here (instead of repeating a
/// multi-clause `all(...)` at ~140 call sites) makes the architecture/feature
/// matrix auditable in ONE place — the exact concern raised in the arm64
/// enablement task ("the cfg generalization must not silently drop a needed
/// gate or mis-route").
///
/// Two aliases:
///
/// * `parallel_sm` — the whole chunked parallel-SM orchestration
///   (chunk_fetcher / chunk_decode / inflate_wrapper / sm_driver / …) is
///   compiled in. True when EITHER:
///     - `x86_64` with `isal-compression` OR `pure-rust-inflate`
///       (x86 historically ran ISA-L; `pure-rust-inflate` is the
///       C-FFI-free path), OR
///     - `aarch64` with `pure-rust-inflate` (NEW: the pure-Rust inner
///       decoder compiles + runs natively on arm64; ISA-L's C library is
///       x86-only and is intentionally NOT required here).
///
/// * `pure_inflate_decode` — the pure-Rust inner decode wrapper
///   (`Inflate<Clean, Generic, Streaming>`) is the active decoder. True on
///   `x86_64` or `aarch64` when `pure-rust-inflate` is enabled. This is the
///   subset of `parallel_sm` that previously read
///   `all(target_arch = "x86_64", feature = "pure-rust-inflate")`.
///
/// Anything that genuinely needs the ISA-L **C** library stays gated on
/// `all(feature = "isal-compression", target_arch = "x86_64")` and is NOT
/// covered by these aliases.
fn emit_parallel_sm_cfgs() {
    // Declare the custom cfgs so `--cfg`/`cfg!()` uses don't trip the
    // `unexpected_cfgs` lint (required since Rust 1.80).
    println!("cargo::rustc-check-cfg=cfg(parallel_sm)");
    println!("cargo::rustc-check-cfg=cfg(pure_inflate_decode)");
    // `isal_clean_tail` is the (now-dormant) ISA-L from-bit clean-tail DECODE
    // oracle selector. It is NEVER emitted: the pure-Rust DEFLATE engine is the
    // SOLE production decode path on every build. The cfg is still DECLARED so
    // the remaining `#[cfg(not(isal_clean_tail))]` branches (the active pure-Rust
    // path) and the dormant `#[cfg(isal_clean_tail)]` branches do not trip the
    // `unexpected_cfgs` lint. The ISA-L from-bit decode survives only as a
    // measurement oracle compiled under `isal-compression` and reachable solely
    // via `GZIPPY_ISAL_ENGINE_ORACLE=1` — off the production decode graph.
    println!("cargo::rustc-check-cfg=cfg(isal_clean_tail)");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let is_x86_64 = arch == "x86_64";
    let is_aarch64 = arch == "aarch64";

    // Cargo exposes enabled features as CARGO_FEATURE_<NAME> (uppercased,
    // `-` → `_`).
    let has_pure_rust_inflate = std::env::var_os("CARGO_FEATURE_PURE_RUST_INFLATE").is_some();

    // The single-member parallel decode (the rapidgzip-shaped path) exists in
    // EXACTLY ONE config: pure-Rust. `isal-compression` only keeps the ISA-L
    // *compression* backend; it does NOT enable any production decode. So the
    // parallel decode is always the pure-Rust decoder, and
    // `pure_inflate_decode == parallel_sm`.
    let parallel_sm = (is_x86_64 || is_aarch64) && has_pure_rust_inflate;
    let pure_inflate_decode = parallel_sm;

    if parallel_sm {
        println!("cargo::rustc-cfg=parallel_sm");
    }
    if pure_inflate_decode {
        println!("cargo::rustc-cfg=pure_inflate_decode");
    }

    // BUILD_FLAVOR: compile-time &'static str consumed by `env!("BUILD_FLAVOR")`.
    //   "parallel-sm+pure"  — parallel_sm (gzippy-native / pure-rust-inflate, the
    //                         pure-Rust DEFLATE engine, x86_64 + aarch64). The product.
    //   "legacy-serial"     — no parallel_sm (a bare no-feature build). NOT the
    //                         product and no longer a working decode build (the
    //                         legacy C-FFI one-shot decode path was removed); a
    //                         `cargo::warning` is emitted below.
    let flavor = if parallel_sm {
        "parallel-sm+pure"
    } else {
        "legacy-serial"
    };
    println!("cargo::rustc-env=BUILD_FLAVOR={flavor}");

    if !parallel_sm {
        println!(
            "cargo::warning=building without the pure-Rust parallel decoder (no parallel_sm) — \
             this build has NO working decode path. \
             Use the default build or `--no-default-features --features gzippy-native`."
        );
    }
}

fn build_zopfli_oracle() {
    let dir = "vendor/zopfli/src/zopfli";
    if !std::path::Path::new(&format!("{}/zopfli.h", dir)).exists() {
        panic!(
            "[oracle] vendor/zopfli is not initialized — run `git submodule \
             update --init vendor/zopfli` before `cargo test --features oracle`"
        );
    }
    println!("cargo:rerun-if-changed={}", dir);
    let mut build = cc::Build::new();
    // macOS: force the system `ar` so `cc` doesn't pick up GNU `ar` from
    // Homebrew, which emits archives the system linker can't read
    // ("ld: archive member '/' not a mach-o file").
    if cfg!(target_os = "macos") && std::path::Path::new("/usr/bin/ar").exists() {
        build.archiver("/usr/bin/ar");
    }
    build
        .include(dir)
        .file(format!("{}/blocksplitter.c", dir))
        .file(format!("{}/cache.c", dir))
        .file(format!("{}/deflate.c", dir))
        .file(format!("{}/gzip_container.c", dir))
        .file(format!("{}/hash.c", dir))
        .file(format!("{}/katajainen.c", dir))
        .file(format!("{}/lz77.c", dir))
        .file(format!("{}/squeeze.c", dir))
        .file(format!("{}/tree.c", dir))
        .file(format!("{}/util.c", dir))
        .file(format!("{}/zlib_container.c", dir))
        .file(format!("{}/zopfli_lib.c", dir))
        .opt_level(2)
        .warnings(false)
        .compile("zopfli_oracle");
}

fn install_git_hooks() {
    // Resolve the *effective* hooks directory via git itself. This works
    // from a linked worktree (where `.git` is a file, not a directory)
    // and honors a custom `core.hooksPath`. The old literal `.git/hooks`
    // silently installed nothing whenever `cargo build` ran inside a
    // worktree, leaving the pre-commit / pre-push hooks frozen at
    // whatever revision was last built from the main checkout.
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--git-path", "hooks"])
        .output();
    let hooks_dir = match output {
        Ok(o) if o.status.success() => {
            std::path::PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string())
        }
        _ => return, // not a git checkout, or git unavailable
    };
    if std::fs::create_dir_all(&hooks_dir).is_err() {
        return;
    }
    install_hook("scripts/pre-commit", &hooks_dir.join("pre-commit"));
    install_hook("scripts/pre-push", &hooks_dir.join("pre-push"));
}

fn install_hook(src: &str, dst: &std::path::Path) {
    let src = std::path::Path::new(src);

    if !src.exists() {
        return;
    }

    // Compare content, not mtime: a worktree checkout gives tracked
    // files arbitrary timestamps, so an mtime heuristic can leave a
    // stale hook installed. Rewrite whenever the bytes differ.
    let content = std::fs::read(src).unwrap_or_else(|_| panic!("read {}", src.display()));
    if std::fs::read(dst).map(|d| d == content).unwrap_or(false) {
        return;
    }
    std::fs::write(dst, &content).unwrap_or_else(|_| panic!("write {}", dst.display()));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dst, std::fs::Permissions::from_mode(0o755))
            .unwrap_or_else(|_| panic!("chmod +x {}", dst.display()));
    }
}
