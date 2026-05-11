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

    #[cfg(feature = "oracle")]
    build_zopfli_oracle();
}

#[cfg(feature = "oracle")]
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
    let git = std::path::Path::new(".git");
    if !git.is_dir() {
        return; // missing or a file (git worktree)
    }
    install_hook("scripts/pre-commit", ".git/hooks/pre-commit");
    install_hook("scripts/pre-push", ".git/hooks/pre-push");
}

fn install_hook(src: &str, dst: &str) {
    let src = std::path::Path::new(src);
    let dst = std::path::Path::new(dst);

    if !src.exists() {
        return;
    }

    let needs_update = !dst.exists() || {
        let src_mtime = src.metadata().and_then(|m| m.modified()).ok();
        let dst_mtime = dst.metadata().and_then(|m| m.modified()).ok();
        src_mtime > dst_mtime
    };

    if needs_update {
        let content = std::fs::read(src).unwrap_or_else(|_| panic!("read {}", src.display()));
        std::fs::write(dst, &content).unwrap_or_else(|_| panic!("write {}", dst.display()));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(dst, std::fs::Permissions::from_mode(0o755))
                .unwrap_or_else(|_| panic!("chmod +x {}", dst.display()));
        }
    }
}
