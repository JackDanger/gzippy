use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionFormat {
    #[default]
    Gzip,
    Zlib,
    Zip,
}

impl CompressionFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "gz" | "gzip" | "tgz" | "taz" | "z" => Some(CompressionFormat::Gzip),
            "zz" => Some(CompressionFormat::Zlib),
            "zip" => Some(CompressionFormat::Zip),
            _ => None,
        }
    }
}

impl fmt::Display for CompressionFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompressionFormat::Gzip => write!(f, "gzip"),
            CompressionFormat::Zlib => write!(f, "zlib"),
            CompressionFormat::Zip => write!(f, "zip"),
        }
    }
}
