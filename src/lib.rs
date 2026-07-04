//! Lightweight mmap-backed exact vector search for [rek0n](https://github.com/K48N/rek0n).

mod compact;
mod ivf;
mod lock;
mod postings;
mod search;
mod storage;
mod types;

pub use lock::{DbLockOptions, DEFAULT_LOCK_TIMEOUT};
pub use search::{dot_product, SearchHit};
pub use storage::Rek0nDb;
pub use types::{
    AnnStrategy, ChunkRecord, CompactionPolicy, CompactionStats, DbError, Point, SearchScope,
    VectorId, DEFAULT_COMPACT_THRESHOLD, DEFAULT_IVF_BUCKETS, DEFAULT_IVF_PROBE, EMBEDDING_DIM,
    MAX_MANIFEST_BYTES, MAX_RECORD_TEXT_BYTES, MAX_STAGING_VECTORS, MAX_VECTORS_BYTES,
};

#[doc(hidden)]
pub mod testing {
    use super::{ChunkRecord, EMBEDDING_DIM};

    pub fn unit_vector(active: usize) -> Vec<f32> {
        let mut vector = vec![0.0_f32; EMBEDDING_DIM];
        vector[active % EMBEDDING_DIM] = 1.0;
        vector
    }

    pub fn chunk_record(file_path: &str, line: u64) -> ChunkRecord {
        ChunkRecord {
            text: format!("chunk at {line}"),
            kind: "Function".into(),
            name: Some("demo".into()),
            file_path: file_path.into(),
            start_line: line,
            end_line: line,
        }
    }

    pub fn chunk_record_with_kind(file_path: &str, kind: &str, line: u64) -> ChunkRecord {
        ChunkRecord {
            text: format!("chunk at {line}"),
            kind: kind.into(),
            name: None,
            file_path: file_path.into(),
            start_line: line,
            end_line: line,
        }
    }
}
