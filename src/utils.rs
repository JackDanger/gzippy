use crate::error::GzippyError;
use std::fs;
use std::path::{Path, PathBuf};

/// Check whether `GZIPPY_DEBUG` is set. Cached after first call.
#[inline]
pub fn debug_enabled() -> bool {
    use std::sync::OnceLock;
    static DEBUG: OnceLock<bool> = OnceLock::new();
    *DEBUG.get_or_init(|| std::env::var("GZIPPY_DEBUG").is_ok())
}

// ── gzip's suffix recognition ────────────────────────────────────────────────

/// The suffix set gzip RECOGNIZES as its own, verbatim from the vendored
/// `get_suffix` `known_suffixes` table (gzip.c:987-991). `-S <suf>` is extra:
/// the C table's first entry is z_suffix itself.
pub const KNOWN_GZIP_SUFFIXES: [&str; 7] = [".gz", ".z", ".taz", ".tgz", "-gz", "-z", "_z"];

/// Mirror of gzip.c `get_suffix` (gzip.c:979-1015): the trailing
/// RECOGNIZED-suffix substring of `name` — original case preserved, because
/// gzip prints it verbatim in the "already has %s suffix -- unchanged"
/// notice (verified: `gzip FOO.GZ` → "already has .GZ suffix -- unchanged")
/// — or None when the name carries no gzip suffix. `ours` is args.suffix
/// (z_suffix) and is checked first, exactly like the C table's `*suf =
/// z_suffix`. Tail matching is case-insensitive (the C code strlwr's the
/// tail buffer) and never matches right after a path separator
/// (gzip.c:1006-1008).
pub fn known_gzip_suffix<'a>(name: &'a str, ours: &str) -> Option<&'a str> {
    let mut candidates: Vec<&str> = Vec::with_capacity(1 + KNOWN_GZIP_SUFFIXES.len());
    candidates.push(ours);
    for s in KNOWN_GZIP_SUFFIXES {
        if s != ours {
            candidates.push(s);
        }
    }
    for suf in candidates {
        let slen = name.len();
        if slen > suf.len() // C "slen > s": name longer than the suffix
            && !name[..slen - suf.len()].ends_with('/') // PATH_SEP guard
            && name[slen - suf.len()..].eq_ignore_ascii_case(suf)
        {
            return Some(&name[slen - suf.len()..]);
        }
    }
    None
}

// ── gzip's input-open contract ───────────────────────────────────────────────

/// How `open_gzip_input` failed, with the PATH GZIP WOULD NAME in the error
/// (`progerror(ifname)` — gzip.c:389-394).
pub enum OpenFailure {
    /// ENOENT all the way through (input absent; for the decompress suffix
    /// ladder, absent under every candidate). `Missing` holds the last-tried
    /// path: the GIVEN name when it already has a recognized suffix
    /// (gzip.c:1103-1106), else the z_suffix-appended name — "No suffix
    /// found, complain using z_suffix" (gzip.c:1147-1151).
    Missing(PathBuf),
    /// Non-ENOENT errno while opening `path` (EACCES, ELOOP, ...). Carries
    /// the path that failed, whose name the rendered message must use
    /// (gzip.c:1104/1151/1136-1140 all `progerror` the *attempted* ifname).
    SysErr(PathBuf, std::io::Error),
}

/// Mirror of gzip.c `open_input_file` (gzip.c:1072-1163) plus its
/// `open_and_stat` helper (gzip.c:1022-1049) — the ENTIRE contract of
/// opening a gzip CLI input, which pre-open exists/is_symink checks could
/// not reproduce:
///
/// - O_NOFOLLOW unless output goes to stdout or -f was typed — gzip "refuse[s]
///   to follow symbolic links unless -c or -f" (gzip.c:1022-1025). A refused
///   link fails `open` with ELOOP and reports like any other errno: "Too many
///   levels of symbolic links", ERROR/exit-1 class — including a DANGLING
///   link, which hits ELOOP before ENOENT (probed: gzip 1.12/1.14 both
///   versions).
/// - In DECOMPRESS mode (`-d`, and `-t`/`-l` set it internally; COMPRESS
///   never retries), a missing `file` WITHOUT a recognized suffix is revealed
///   against the gzip suffix ladder `{z_suffix, .gz, .z, -z, .Z}`
///   (gzip.c:1077-1079, 1100-1146) — the archive may live under its suffixed
///   name, and the RESOLVED path then replaces the operand for all
///   downstream naming.
pub fn open_gzip_input(
    path: &Path,
    decompress: bool,
    stdout_mode: bool,
    force: bool,
    suffix: &str,
) -> Result<(fs::File, PathBuf), OpenFailure> {
    match open_once(path, stdout_mode, force) {
        Ok(f) => return Ok((f, path.to_path_buf())),
        // COMPRESS mode has no retry (gzip.c:1100-1101); its missing-file
        // error names the given path directly.
        Err(e) if !decompress => {
            return Err(match e.kind() {
                std::io::ErrorKind::NotFound => OpenFailure::Missing(path.to_path_buf()),
                _ => OpenFailure::SysErr(path.to_path_buf(), e),
            })
        }
        // DECOMPRESS mode + ENOENT: fall through to the suffix ladder below.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(OpenFailure::SysErr(path.to_path_buf(), e)),
    }

    // For decompression: `file.ext` doesn't exist, try adding the suffixes
    // (gzip.c:1108 comment). If the given name already carries a recognized
    // suffix, gzip does NOT retry (gzip.c:1103-1106: "ifname already has z
    // suffix and does not exist") — the missing-file error names it directly.
    if let Some(name) = path.to_str() {
        if known_gzip_suffix(name, suffix).is_some() {
            return Err(OpenFailure::Missing(path.to_path_buf()));
        }
    }

    // gzip.c:1077-1079: suffixes[] = {NULL, ".gz", ".z", "-z", ".Z", NULL};
    // z_suffix is tried FIRST and is what the final complaint names.
    let mut ladder: Vec<&str> = vec![suffix];
    for s in [".gz", ".z", "-z", ".Z"] {
        if !ladder.contains(&s) {
            ladder.push(s);
        }
    }

    let mut missing_suffix_name: Option<PathBuf> = None;
    for suf in &ladder {
        let attempt = path_with_suffix(path, suf);
        match open_once(&attempt, stdout_mode, force) {
            Ok(f) => return Ok((f, attempt)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // The complaint names the z_suffix-appended attempt
                // (gzip.c:1147-1151), so remember the FIRST candidate only.
                if missing_suffix_name.is_none() {
                    missing_suffix_name = Some(attempt);
                }
            }
            // gzip.c:1136-1140: a non-ENOENT errno exits the ladder and is
            // reported against the attempted (suffixed) ifname.
            Err(e) => return Err(OpenFailure::SysErr(attempt, e)),
        }
    }
    Err(OpenFailure::Missing(
        missing_suffix_name.unwrap_or_else(|| path.to_path_buf()),
    ))
}

/// One `open()` attempt with gzip's flags (gzip.c:1119-1121): O_NOFOLLOW
/// outside -c/-f, and O_NONBLOCK | O_NOCTTY like `open_input_file` — a FIFO
/// input opened read-only WITHOUT O_NONBLOCK would block forever when the
/// writer side does not exist (gzip never tries to block: the treat_file
/// ladder skips non-regular files before any read).
fn open_once(path: &Path, stdout_mode: bool, force: bool) -> std::io::Result<fs::File> {
    let mut opts = fs::OpenOptions::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // gzip.c:1119-1121: O_RDONLY | O_NONBLOCK | O_NOCTTY — a bare FIFO
        // (writer open pending, or none) must not block the open.
        let mut flags = libc::O_NONBLOCK | libc::O_NOCTTY;
        // gzip "refuse[s] to follow symbolic links unless -c or -f": a
        // refused link (dangling or not) fails open() with ELOOP, which is
        // how the refusal reaches the caller as an errno (and then renders
        // strerror-accurately through OpenFailure::SysErr).
        if !stdout_mode && !force {
            flags |= libc::O_NOFOLLOW;
        }
        opts.custom_flags(flags);
    }
    #[cfg(not(unix))]
    let _ = (stdout_mode, force);
    opts.read(true).open(path)
}

/// `strcat(ifname, z_suffix)` — plain concatenation, no separator inserted
/// (gzip.c:1128/1149), so "file" + ".gz" → "file.gz".
fn path_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.to_string_lossy().into_owned();
    s.push_str(suffix);
    PathBuf::from(s)
}

/// Report an [`OpenFailure`] the way gzip does and return its exit class.
/// `progerror` (gzip.c:389-394) prints `program: name: strerror(errno)` and
/// sets `exit_code = ERROR` (1) for THIS file while the invocation's
/// remaining files keep processing (treat_file just returns) — which is
/// exactly main.rs's `bump_exit_code` ledger fed `1`.
pub fn report_open_failure(failure: OpenFailure, quiet: bool) -> i32 {
    let (path, e) = match failure {
        OpenFailure::Missing(p) => (p, std::io::Error::from_raw_os_error(libc::ENOENT)),
        OpenFailure::SysErr(p, e) => (p, e),
    };
    if !quiet {
        eprintln!("gzippy: {}: {}", path.display(), GzippyError::os(&e));
    }
    1
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

/// gzip's real exit-code precedence, established by execution (gzip 1.14):
/// ERROR (1) ALWAYS wins regardless of arrival order; WARNING (2) — e.g. "is
/// a directory -- ignored" — only takes the code if nothing worse has been
/// recorded yet. A multi-file invocation with `good1 baddir missing2 good2`
/// still processes both good files AND settles on exit 1 (error), never 2,
/// regardless of whether the directory-warning or the missing-file-error
/// argument came first — verified both orders give exit 1 on real gzip.
///
/// Mechanism, from the vendored source: ERROR is set unconditionally
/// (`exit_code = ERROR;` — gzip.c:1165/1324/1476/1815...), while warnings go
/// through WARN which escalates ONLY from OK: "if (exit_code == OK)
/// exit_code = WARNING;" (gzip.h:263-264). Mirrors of that escalation for
/// the -r directory walks live in compress_directory/decompress_directory.
pub fn bump_exit_code(exit_code: &mut i32, new_code: i32) {
    if new_code == 1 {
        *exit_code = 1;
    } else if new_code != 0 && *exit_code == 0 {
        *exit_code = new_code;
    }
}
