use std::fmt;
use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum GzippyError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("Decompression error: {0}")]
    Decompression(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Invalid compression level: {0}")]
    InvalidLevel(u8),

    #[error("Invalid block size: {0}")]
    InvalidBlockSize(String),

    /// Pure `strerror(3)` text ("No such file or directory", "Permission
    /// denied", "Too many levels of symbolic links"). gzip's contract for
    /// open/stat failures is `progerror` (gzip.c:389-394): the stderr line is
    /// `program: pathname: strerror(errno)` — the operating system's errno
    /// text, never a gzippy-specific wording and never a duplicated pathname.
    /// Built with [`GzippyError::os`].
    #[error("{0}")]
    Sys(String),

    /// `get_method` (gzip.c:1473-1475): `"\n%s: %s: not in gzip format\n"` —
    /// ERROR class, exit 1. The leading blank line is part of gzip's
    /// observable stderr (separating the message from any partial output);
    /// renderers see it via [`GzippyError::has_leading_blank_line`].
    #[error("not in gzip format")]
    NotInGzipFormat,

    /// `read_error` (util.c:394-402), which gzip ties to EOF before/inside
    /// the gzip magic: `"\ngzip: NAME: unexpected end of file"` — same
    /// leading-blank-line contract as [`GzippyError::NotInGzipFormat`].
    #[error("unexpected end of file")]
    UnexpectedEof,

    /// GNU getopt's usage-error framing for an unparseable command line.
    /// getopt_long emits its own message and `try_help` (gzip.c:315-319)
    /// prints `Try 'gzip --help' for more information.` then exits ERROR
    /// (do_exit(ERROR) — gzip.1's documented "if an error occurs, exit
    /// status is 1"). The payload is getopt's line itself ("invalid option
    /// -- 'Q'", "option requires an argument -- 'S'", "option '--suffix'
    /// requires an argument"); main's `gzippy: ` prefix + newline + the Try
    /// line reproduce gzip's two-line stderr exactly.
    #[error("{0}\nTry `gzippy --help' for more information.")]
    Usage(String),

    #[error("Thread error: {0}")]
    Thread(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("WalkDir error: {0}")]
    WalkDir(#[from] walkdir::Error),
}

#[allow(dead_code)]
impl GzippyError {
    pub fn compression<T: fmt::Display>(msg: T) -> Self {
        GzippyError::Compression(msg.to_string())
    }

    /// The strerror(3) text of an OS error, as gzip renders it via `perror`
    /// in `progerror` (gzip.c:389-394): glibc's `No such file or directory`,
    /// NOT Rust's `No such file or directory (os error 2)` rendering.
    pub fn os(e: &io::Error) -> Self {
        let text = match e.raw_os_error() {
            Some(code) => {
                // SAFETY: `strerror` always accepts an arbitrary errno value;
                // glibc maps unknown codes to a thread-local "Unknown error N"
                // string that stays valid long enough for this copy.
                let raw = unsafe { libc::strerror(code) };
                if raw.is_null() {
                    e.to_string()
                } else {
                    // SAFETY: checked non-null above; NUL-terminated and valid
                    // until the next strerror call.
                    unsafe { std::ffi::CStr::from_ptr(raw) }
                        .to_string_lossy()
                        .into_owned()
                }
            }
            None => e.to_string(), // non-OS io::Error; best effort
        };
        GzippyError::Sys(text)
    }

    pub fn decompression<T: fmt::Display>(msg: T) -> Self {
        GzippyError::Decompression(msg.to_string())
    }

    pub fn invalid_argument<T: fmt::Display>(msg: T) -> Self {
        GzippyError::InvalidArgument(msg.to_string())
    }

    /// get_method (gzip.c:1474) and read_error (util.c:397/400) print these
    /// two messages with a LEADING blank line before "program: name:". The
    /// per-file error renderers in main/decompress consult this so the blank
    /// line stays part of the observable stream.
    pub fn has_leading_blank_line(&self) -> bool {
        matches!(
            self,
            GzippyError::NotInGzipFormat | GzippyError::UnexpectedEof
        )
    }

    pub fn thread<T: fmt::Display>(msg: T) -> Self {
        GzippyError::Thread(msg.to_string())
    }

    pub fn internal<T: fmt::Display>(msg: T) -> Self {
        GzippyError::Internal(msg.to_string())
    }

    pub fn parse<T: fmt::Display>(msg: T) -> Self {
        GzippyError::Parse(msg.to_string())
    }
}

pub type GzippyResult<T> = Result<T, GzippyError>;
