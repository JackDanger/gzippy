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

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

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

    pub fn decompression<T: fmt::Display>(msg: T) -> Self {
        GzippyError::Decompression(msg.to_string())
    }

    pub fn invalid_argument<T: fmt::Display>(msg: T) -> Self {
        GzippyError::InvalidArgument(msg.to_string())
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
