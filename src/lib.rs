//! Lightweight mmap-backed exact vector search for [rek0n](https://github.com/K48N/rek0n).
//!
//! `rek0n-db` is the "SQLite of vector databases": append-only `f32` vectors,
//! JSON manifest sidecar, tombstones with lazy compaction, inverted posting
//! lists for scoped search, and an optional IVF-lite ANN tier.
//!
//! # Two-tier architecture
//!
//! - **Persistent tier** — memory-mapped append-only `vectors.bin`.
//! - **Staging tier** — in-memory vectors for MCTS / ephemeral branches.
//!
//! # Search tiers
//!
//! - **Tier 0** — exact dot-product over all live vectors.
//! - **Tier 1** — exact search restricted by [`SearchScope`] posting filters.
//! - **Tier 2** — [`AnnStrategy::Ivf`] probes nearest centroid buckets first.
//! - **Tier 3** — [`AnnStrategy::Hnsw`] reserved for future `rek0n-search` integration.

mod compact;
mod ivf;
mod postings;
mod search;
mod storage;
mod types;

pub use search::{dot_product, SearchHit};
pub use storage::Rek0nDb;
pub use types::{
    AnnStrategy, ChunkRecord, CompactionPolicy, CompactionStats, DbError, Point, SearchScope,
    VectorId, DEFAULT_COMPACT_THRESHOLD, DEFAULT_IVF_BUCKETS, DEFAULT_IVF_PROBE, EMBEDDING_DIM,
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
