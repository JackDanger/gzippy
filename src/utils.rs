use std::fs;
use std::path::Path;

/// Check whether `GZIPPY_DEBUG` is set. Cached after first call.
#[inline]
pub fn debug_enabled() -> bool {
    use std::sync::OnceLock;
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var("GZIPPY_DEBUG").is_ok())
}

/// Copy permissions and mtime from `src` to `dst`.
/// Errors are silently ignored — best-effort, matching gzip behavior.
pub fn preserve_metadata(src: &Path, dst: &Path) {
    if let Ok(metadata) = fs::metadata(src) {
        let _ = fs::set_permissions(dst, metadata.permissions());
        if let Ok(mtime) = metadata.modified() {
            let _ = filetime::set_file_mtime(dst, filetime::FileTime::from_system_time(mtime));
        }
    }
}

pub fn detect_format_from_file(path: &Path) -> Option<crate::format::CompressionFormat> {
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            return crate::format::CompressionFormat::from_extension(ext_str);
        }
    }
    None
}

pub fn is_compressed_file(path: &Path) -> bool {
    detect_format_from_file(path).is_some()
}

pub fn strip_compression_extension(path: &Path) -> std::path::PathBuf {
    let mut result = path.to_path_buf();

    // Handle special suffix mappings first
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            match ext_str.to_lowercase().as_str() {
                // .tgz → .tar, .taz → .tar (standard gzip convention)
                "tgz" | "taz" => {
                    result.set_extension("tar");
                    return result;
                }
                _ => {}
            }
            if crate::format::CompressionFormat::from_extension(ext_str).is_some() {
                result.set_extension("");
            }
        }
    }

    // Also handle -gz, -z, _z suffixes (e.g., file-gz → file)
    if result == path.to_path_buf() {
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            let lower = name.to_lowercase();
            for suffix in &["-gz", "-z", "_z"] {
                if lower.ends_with(suffix) {
                    let new_name = &name[..name.len() - suffix.len()];
                    result.set_file_name(new_name);
                    return result;
                }
            }
        }
    }

    result
}
