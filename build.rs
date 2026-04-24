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
    install_git_hooks();
}

fn install_git_hooks() {
    let hook_src = std::path::Path::new("scripts/pre-commit");
    let hook_dst = std::path::Path::new(".git/hooks/pre-commit");

    let git_path = std::path::Path::new(".git");
    if !hook_src.exists() || !git_path.exists() || !git_path.is_dir() {
        return; // skip in git worktrees (.git is a file, not a dir)
    }

    // Overwrite if missing or stale (src newer than dst)
    let needs_update = !hook_dst.exists() || {
        let src_mtime = hook_src.metadata().and_then(|m| m.modified()).ok();
        let dst_mtime = hook_dst.metadata().and_then(|m| m.modified()).ok();
        src_mtime > dst_mtime
    };

    if needs_update {
        let content = std::fs::read(hook_src).expect("read scripts/pre-commit");
        std::fs::write(hook_dst, content).expect("write .git/hooks/pre-commit");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(hook_dst, std::fs::Permissions::from_mode(0o755))
                .expect("chmod +x .git/hooks/pre-commit");
        }
    }
}
