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
    println!("cargo:rerun-if-changed=vendor/zopfli/src");
    compile_zopfli();
    install_git_hooks();
}

fn compile_zopfli() {
    let src_dir = "vendor/zopfli/src/zopfli";
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = std::path::Path::new(&out_dir);

    let sources = [
        "blocksplitter.c",
        "cache.c",
        "deflate.c",
        "gzip_container.c",
        "hash.c",
        "katajainen.c",
        "lz77.c",
        "squeeze.c",
        "tree.c",
        "util.c",
        "zlib_container.c",
        "zopfli_lib.c",
    ];

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());

    let mut obj_files = Vec::new();
    for source in &sources {
        let obj_file = out_path.join(format!("zopfli-{}.o", source.trim_end_matches(".c")));
        let src_file = format!("{}/{}", src_dir, source);

        let output = std::process::Command::new(&cc)
            .arg("-c")
            .arg("-O3")
            .arg("-fPIC")
            .arg(format!("-I{}", src_dir))
            .arg(&src_file)
            .arg("-o")
            .arg(&obj_file)
            .output()
            .unwrap_or_else(|_| panic!("Failed to compile {}", source));

        if !output.status.success() {
            panic!(
                "Failed to compile {}: {}",
                source,
                String::from_utf8_lossy(&output.stderr)
            );
        }
        obj_files.push(obj_file);
    }

    let lib_file = out_path.join("libzopfli.a");
    let mut ar_cmd = std::process::Command::new("ar");
    ar_cmd.arg("rcs").arg(&lib_file);
    for obj in &obj_files {
        ar_cmd.arg(obj);
    }

    let output = ar_cmd.output().expect("Failed to run ar");
    if !output.status.success() {
        panic!(
            "Failed to create archive: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    println!("cargo:rustc-link-lib=static=zopfli");
    println!("cargo:rustc-link-search={}", out_dir);
}

fn install_git_hooks() {
    let git = std::path::Path::new(".git");
    if !git.is_dir() {
        return; // missing or a file (git worktree)
    }
    install_hook("scripts/pre-commit", ".git/hooks/pre-commit");
    install_hook("scripts/pre-push", ".git/hooks/pre-push");
}

fn compile_zopfli() {
    let src_dir = "vendor/zopfli/src/zopfli";
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = std::path::Path::new(&out_dir);

    let sources = [
        "blocksplitter.c",
        "cache.c",
        "deflate.c",
        "gzip_container.c",
        "hash.c",
        "katajainen.c",
        "lz77.c",
        "squeeze.c",
        "tree.c",
        "util.c",
        "zlib_container.c",
        "zopfli_lib.c",
    ];

    // Compile each file to .o
    let mut obj_files = Vec::new();
    for source in &sources {
        let obj_file = out_path.join(format!("zopfli-{}.o", source.trim_end_matches(".c")));
        let src_file = format!("{}/{}", src_dir, source);

        let output = std::process::Command::new("clang")
            .arg("-c")
            .arg("-O3")
            .arg("-fPIC")
            .arg(format!("-I{}", src_dir))
            .arg(&src_file)
            .arg("-o")
            .arg(&obj_file)
            .output()
            .unwrap_or_else(|_| panic!("Failed to compile {}", source));

        if !output.status.success() {
            panic!(
                "Failed to compile {}: {}",
                source,
                String::from_utf8_lossy(&output.stderr)
            );
        }
        obj_files.push(obj_file);
    }

    // Create archive with libtool
    let lib_file = out_path.join("libzopfli.a");
    let mut libtool_cmd = std::process::Command::new("libtool");
    libtool_cmd.arg("-static").arg("-o").arg(&lib_file);
    for obj in &obj_files {
        libtool_cmd.arg(obj);
    }

    let output = libtool_cmd.output().expect("Failed to run libtool");
    if !output.status.success() {
        panic!(
            "Failed to create archive: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // Link
    println!("cargo:rustc-link-lib=static=zopfli");
    println!("cargo:rustc-link-search={}", out_dir);
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
