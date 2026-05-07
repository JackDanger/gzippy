//! --index and --seek modes for indexed/random-access decompression.

use crate::cli::GzippyArgs;
use std::io;
use std::path::Path;

pub fn maybe_run(args: &GzippyArgs) -> Option<i32> {
    if args.build_index || args.seek.is_some() {
        Some(run_index_mode(args))
    } else {
        None
    }
}

fn run_index_mode(args: &GzippyArgs) -> i32 {
    if args.build_index && args.seek.is_some() {
        eprintln!("gzippy: --index and --seek cannot be used together");
        return 1;
    }

    if args.build_index {
        return run_build_index(args);
    }

    if let Some(offset) = args.seek {
        return run_seek_decompress(args, offset);
    }

    eprintln!("gzippy: internal error: index_mode activated but no action selected");
    1
}

fn run_build_index(args: &GzippyArgs) -> i32 {
    let input_path = match args.files.first() {
        Some(path) if path != "-" => path.clone(),
        _ => {
            eprintln!("gzippy: --index requires an input file (not stdin)");
            return 1;
        }
    };

    // Load the input file
    let gzip_data = match std::fs::read(&input_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("gzippy: failed to read {}: {}", input_path, e);
            return 1;
        }
    };

    // Build the index
    let index = match crate::decompress::index::build_index(&gzip_data, args.index_interval) {
        Ok(idx) => idx,
        Err(e) => {
            eprintln!("gzippy: failed to build index: {}", e);
            return 1;
        }
    };

    // Determine output path
    let output_path = args
        .index_file
        .as_ref()
        .cloned()
        .unwrap_or_else(|| format!("{}.gzidx", input_path));

    // Serialize the index
    match std::fs::File::create(&output_path) {
        Ok(mut file) => {
            if let Err(e) = crate::decompress::index::serialize_index(&index, &mut file) {
                eprintln!("gzippy: failed to serialize index: {}", e);
                let _ = std::fs::remove_file(&output_path);
                return 1;
            }
        }
        Err(e) => {
            eprintln!("gzippy: failed to create index file {}: {}", output_path, e);
            return 1;
        }
    }

    eprintln!(
        "gzippy: built index with {} checkpoints → {}",
        index.points.len(),
        output_path
    );
    0
}

fn run_seek_decompress(args: &GzippyArgs, offset: u64) -> i32 {
    let input_path = match args.files.first() {
        Some(path) if path != "-" => path.clone(),
        _ => {
            eprintln!("gzippy: --seek requires an input file (not stdin)");
            return 1;
        }
    };

    // Check that output is to stdout
    if !args.stdout {
        eprintln!("gzippy: --seek requires -c or --stdout");
        return 1;
    }

    // Load the input file
    let gzip_data = match std::fs::read(&input_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("gzippy: failed to read {}: {}", input_path, e);
            return 1;
        }
    };

    // Determine index path
    let index_path = args
        .index_file
        .as_ref()
        .cloned()
        .unwrap_or_else(|| format!("{}.gzidx", input_path));

    // Load or build the index
    let index = if Path::new(&index_path).exists() {
        match std::fs::read(&index_path) {
            Ok(data) => match crate::decompress::index::load_index(&data) {
                Ok(idx) => idx,
                Err(e) => {
                    eprintln!("gzippy: failed to load index {}: {}", index_path, e);
                    return 1;
                }
            },
            Err(e) => {
                eprintln!("gzippy: failed to read index file {}: {}", index_path, e);
                return 1;
            }
        }
    } else {
        eprintln!(
            "gzippy: index file {} not found. Build it with: gzippy --index {}",
            index_path, input_path
        );
        return 1;
    };

    // Decompress from the seek offset
    let stdout = io::stdout();
    let mut stdout_lock = stdout.lock();
    let mut writer = io::BufWriter::with_capacity(1024 * 1024, &mut stdout_lock);

    match crate::decompress::index::seek_decompress(
        &gzip_data,
        &index,
        offset,
        u64::MAX,
        &mut writer,
    ) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("gzippy: seek decompress failed: {}", e);
            1
        }
    }
}
