//! Build script for gzippy
//!
//! gzippy uses only statically-linked dependencies for safe distribution:
//! - libdeflate (via libdeflater crate) - statically linked, highly optimized
//! - zlib-ng (via flate2 with zlib-ng feature) - statically linked
//!
//! Note: ISA-L FFI was attempted but the complex struct layout (200KB+ with
//! nested Huffman tables) makes Rust FFI difficult. Since ISA-L uses pure C
//! fallback on ARM anyway, libdeflate is the pragmatic choice.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=scripts/pre-commit");
    println!("cargo:rerun-if-changed=scripts/pre-push");
    install_git_hooks();
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
