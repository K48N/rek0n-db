use serde::{Deserialize, Serialize};

/// Stable row identifier across persistent and staging tiers.
pub type VectorId = u32;

/// Default embedding width used by rek0n-embed (all-MiniLM-L6-v2).
pub const EMBEDDING_DIM: usize = 384;

/// Default IVF bucket count for Tier-2 search.
pub const DEFAULT_IVF_BUCKETS: usize = 256;

/// Default number of centroid buckets to probe during IVF search.
pub const DEFAULT_IVF_PROBE: usize = 3;

/// Default dead-vector ratio before lazy compaction runs.
pub const DEFAULT_COMPACT_THRESHOLD: f32 = 0.25;

/// Maximum `manifest.json` size accepted at open.
pub const MAX_MANIFEST_BYTES: usize = 64 * 1024 * 1024;

/// Maximum metadata text length stored per chunk record.
pub const MAX_RECORD_TEXT_BYTES: usize = 262_144;

/// Maximum `vectors.bin` size accepted at open.
pub const MAX_VECTORS_BYTES: u64 = 512 * 1024 * 1024;

/// Maximum vectors held in the in-memory staging tier before flush.
#[cfg(test)]
pub const MAX_STAGING_VECTORS: usize = 4;
#[cfg(not(test))]
pub const MAX_STAGING_VECTORS: usize = 100_000;

/// A single indexed vector with its external metadata key.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    pub id: VectorId,
    pub vector: Vec<f32>,
    pub record: ChunkRecord,
}

/// Serializable chunk metadata for rek0n-embed integration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkRecord {
    pub text: String,
    pub kind: String,
    pub name: Option<String>,
    pub file_path: String,
    pub start_line: u64,
    pub end_line: u64,
}

impl ChunkRecord {
    pub fn metadata_id(&self) -> String {
        format!("{}:{}:{}", self.file_path, self.start_line, self.end_line)
    }
}

/// Structured search filter — SQL-style predicates without SQL.
///
/// Combine filters freely. When every field is `None` and `include_staging` is
/// true, search scans all live vectors (Tier 0 exact).
#[derive(Debug, Clone, Default)]
pub struct SearchScope<'a> {
    /// Restrict to an explicit set of repository paths.
    pub file_paths: Option<&'a [String]>,
    /// Restrict to paths beginning with this prefix (`src/auth/`).
    pub file_path_prefix: Option<&'a str>,
    /// Restrict to chunk kinds (`Function`, `Struct`, …).
    pub kinds: Option<&'a [String]>,
    /// Restrict to an explicit candidate id set (GraphRAG / MCTS neighborhoods).
    pub candidate_ids: Option<&'a [VectorId]>,
    /// Include the in-memory staging tier (default: true).
    pub include_staging: bool,
}

impl SearchScope<'_> {
    pub fn all() -> Self {
        Self {
            include_staging: true,
            ..Default::default()
        }
    }

    pub fn is_unrestricted(&self) -> bool {
        self.file_paths.is_none()
            && self.file_path_prefix.is_none()
            && self.kinds.is_none()
            && self.candidate_ids.is_none()
    }
}

/// ANN tier selection for search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnnStrategy {
    /// Tier 0 — exact dot product over all live vectors (or scoped candidates).
    #[default]
    Exact,
    /// Tier 2 — IVF-lite: probe nearest centroid buckets, exact search within union.
    Ivf { probe_buckets: usize },
    /// Tier 3 — HNSW graph search (reserved; requires future `rek0n-search` crate).
    Hnsw { ef_search: usize },
}

/// Lazy compaction policy (tombstones accumulate until threshold).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompactionPolicy {
    /// Compact when `tombstones / allocated_ids >= threshold`.
    pub dead_ratio_threshold: f32,
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            dead_ratio_threshold: DEFAULT_COMPACT_THRESHOLD,
        }
    }
}

impl CompactionPolicy {
    pub fn validate(self) -> Result<Self, DbError> {
        if !self.dead_ratio_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.dead_ratio_threshold)
        {
            return Err(DbError::InvalidCompactionPolicy {
                threshold: self.dead_ratio_threshold,
            });
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionStats {
    pub vectors_before: usize,
    pub vectors_after: usize,
    pub bytes_reclaimed: u64,
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DbError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("memory map error: {0}")]
    Mmap(String),

    #[error("invalid vector dimension: expected {expected}, got {got}")]
    InvalidDimension { expected: usize, got: usize },

    #[error("vector data is not f32-aligned (byte length {len})")]
    UnalignedVectorData { len: usize },

    #[error("vector data length {len} is not a multiple of f32 size")]
    InvalidVectorBytes { len: usize },

    #[error("metadata missing for vector id {id}")]
    MissingMetadata { id: VectorId },

    #[error("query vector must be {expected}-dimensional, got {got}")]
    InvalidQuery { expected: usize, got: usize },

    #[error("search limit must be > 0")]
    InvalidSearchLimit,

    #[error("IVF index not built — call build_ivf_index() first")]
    IvfNotBuilt,

    #[error("HNSW index not built — rek0n-search is not wired yet")]
    HnswNotBuilt,

    #[error("not enough live vectors ({live}) to build {buckets} IVF buckets")]
    InsufficientVectorsForIvf { live: usize, buckets: usize },

    #[error("vector offset missing for id {id}")]
    MissingOffset { id: VectorId },

    #[error("IVF bucket count must be greater than zero")]
    InvalidIvfBucketCount,

    #[error("corrupt manifest: invalid vector id {value:?}")]
    InvalidManifestId { value: String },

    #[error("replace_file: record file_path {got:?} does not match target {expected:?}")]
    FilePathMismatch { expected: String, got: String },

    #[error("vector id space exhausted")]
    IdExhausted,

    #[error("manifest is {len} bytes, exceeding limit of {max}")]
    ManifestTooLarge { len: usize, max: usize },

    #[error("vectors.bin is {len} bytes, exceeding limit of {max}")]
    VectorsFileTooLarge { len: u64, max: u64 },

    #[error("unsupported manifest version {got}, expected {expected}")]
    UnsupportedManifestVersion { got: u32, expected: u32 },

    #[error("corrupt manifest: vector offset for id {id} extends past vectors.bin")]
    CorruptVectorOffset { id: VectorId },

    #[error("database lock timed out at {path}")]
    LockTimeout { path: String },

    #[error("database opened read-only")]
    ReadOnly,

    #[error("record text is {len} bytes, exceeding limit of {max}")]
    RecordTextTooLarge { len: usize, max: usize },

    #[error("compaction dead_ratio_threshold {threshold} must be finite and in [0.0, 1.0]")]
    InvalidCompactionPolicy { threshold: f32 },

    #[error("staging tier holds {count} vectors, exceeding limit of {max}")]
    StagingCapacityExceeded { count: usize, max: usize },

    #[error("cannot change embedding dimension on a non-empty database")]
    DimensionChangeOnNonEmptyDb,

    #[error("corrupt manifest: records and offsets key sets differ")]
    CorruptManifestKeyMismatch,

    #[error("corrupt manifest: tombstone id {id} has no record")]
    CorruptManifestTombstone { id: VectorId },

    #[error("corrupt manifest: duplicate vector offset {offset} for ids {first} and {second}")]
    DuplicateVectorOffset {
        offset: u64,
        first: VectorId,
        second: VectorId,
    },

    #[error("corrupt manifest: vectors.bin size {file_len} is smaller than required {required}")]
    VectorsFileTooSmall { file_len: usize, required: usize },

    #[error("corrupt centroids.bin: {reason}")]
    CorruptCentroids { reason: String },
}

impl DbError {
    pub fn io_path(path: impl AsRef<std::path::Path>, source: std::io::Error) -> Self {
        DbError::Io {
            path: path.as_ref().display().to_string(),
            source,
        }
    }
}
