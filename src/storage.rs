use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};

use crate::compact::{compact_vectors, dead_ratio};
use crate::ivf::{
    build_ivf_index, centroids_path, default_ivf_buckets, default_ivf_probe, read_centroids,
    write_centroids, IvfIndex,
};
use crate::lock::{DbLock, DbLockOptions, DEFAULT_LOCK_TIMEOUT};
use crate::postings::PostingIndex;
use crate::types::{
    ChunkRecord, CompactionPolicy, CompactionStats, DbError, Point, VectorId, EMBEDDING_DIM,
    MAX_MANIFEST_BYTES, MAX_RECORD_TEXT_BYTES, MAX_STAGING_VECTORS, MAX_VECTORS_BYTES,
};

const VECTORS_FILE: &str = "vectors.bin";
const MANIFEST_FILE: &str = "manifest.json";
const MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    dim: usize,
    next_id: VectorId,
    compaction: CompactionPolicy,
    records: HashMap<String, ChunkRecord>,
    offsets: HashMap<String, u64>,
    tombstones: Vec<VectorId>,
    postings: PostingIndex,
    ivf: Option<IvfManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IvfManifest {
    num_buckets: usize,
    probe_buckets: usize,
    assignments: HashMap<String, usize>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            version: MANIFEST_VERSION,
            dim: EMBEDDING_DIM,
            next_id: 0,
            compaction: CompactionPolicy::default(),
            records: HashMap::new(),
            offsets: HashMap::new(),
            tombstones: Vec::new(),
            postings: PostingIndex::default(),
            ivf: None,
        }
    }
}

/// Two-tier vector store: append-only mmap persistent tier + in-memory staging.
pub struct Rek0nDb {
    dir: PathBuf,
    dim: usize,
    mmap: Mmap,
    next_id: VectorId,
    records: HashMap<VectorId, ChunkRecord>,
    id_offsets: HashMap<VectorId, u64>,
    tombstones: HashSet<VectorId>,
    postings: PostingIndex,
    ivf: Option<IvfIndex>,
    compaction_policy: CompactionPolicy,
    staging_vectors: Vec<f32>,
    staging_records: HashMap<VectorId, ChunkRecord>,
    staging_order: Vec<VectorId>,
    read_only: bool,
    _lock: DbLock,
}

impl Rek0nDb {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, DbError> {
        Self::open_with_options(dir, DbLockOptions::default())
    }

    pub fn open_read_only(dir: impl AsRef<Path>) -> Result<Self, DbError> {
        Self::open_with_options(dir, DbLockOptions::shared(DEFAULT_LOCK_TIMEOUT))
    }

    pub fn open_with_options(dir: impl AsRef<Path>, lock: DbLockOptions) -> Result<Self, DbError> {
        let read_only = lock.read_only();
        let dir = dir.as_ref().to_path_buf();
        let db_lock = DbLock::acquire(&dir, lock)?;
        Self::open_locked(dir, read_only, db_lock)
    }

    fn open_locked(dir: PathBuf, read_only: bool, db_lock: DbLock) -> Result<Self, DbError> {
        fs::create_dir_all(&dir).map_err(|source| DbError::io_path(&dir, source))?;

        let vectors_path = dir.join(VECTORS_FILE);
        if !vectors_path.exists() {
            File::create(&vectors_path)
                .map_err(|source| DbError::io_path(&vectors_path, source))?;
        }

        let manifest_path = dir.join(MANIFEST_FILE);
        let manifest = if manifest_path.exists() {
            let bytes = fs::read(&manifest_path)
                .map_err(|source| DbError::io_path(&manifest_path, source))?;
            if bytes.len() > MAX_MANIFEST_BYTES {
                return Err(DbError::ManifestTooLarge {
                    len: bytes.len(),
                    max: MAX_MANIFEST_BYTES,
                });
            }
            let manifest: Manifest = serde_json::from_slice(&bytes)?;
            if manifest.version != MANIFEST_VERSION {
                return Err(DbError::UnsupportedManifestVersion {
                    got: manifest.version,
                    expected: MANIFEST_VERSION,
                });
            }
            manifest
        } else {
            let default = Manifest::default();
            write_manifest(&manifest_path, &default)?;
            default
        };

        let mmap = map_vectors_file(&vectors_path)?;
        if mmap.len() as u64 > MAX_VECTORS_BYTES {
            return Err(DbError::VectorsFileTooLarge {
                len: mmap.len() as u64,
                max: MAX_VECTORS_BYTES,
            });
        }
        let records = decode_record_map(&manifest.records)?;
        let id_offsets = decode_offset_map(&manifest.offsets)?;
        let tombstones: HashSet<VectorId> = manifest.tombstones.into_iter().collect();
        validate_open_state(manifest.dim, &records, &id_offsets, &tombstones, mmap.len())?;
        let ivf = load_ivf(&dir, manifest.dim, manifest.ivf)?;

        Ok(Self {
            dir,
            dim: manifest.dim,
            mmap,
            next_id: manifest.next_id,
            records,
            id_offsets,
            tombstones,
            postings: manifest.postings,
            ivf,
            compaction_policy: manifest.compaction.validate()?,
            staging_vectors: Vec::new(),
            staging_records: HashMap::new(),
            staging_order: Vec::new(),
            read_only,
            _lock: db_lock,
        })
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    fn ensure_writable(&self) -> Result<(), DbError> {
        if self.read_only {
            return Err(DbError::ReadOnly);
        }
        Ok(())
    }

    pub fn with_dim(mut self, dim: usize) -> Result<Self, DbError> {
        if dim == 0 {
            return Err(DbError::InvalidDimension {
                expected: EMBEDDING_DIM,
                got: 0,
            });
        }
        if dim != self.dim && !self.is_empty() {
            return Err(DbError::DimensionChangeOnNonEmptyDb);
        }
        self.dim = dim;
        Ok(self)
    }

    pub fn with_compaction_policy(mut self, policy: CompactionPolicy) -> Result<Self, DbError> {
        self.compaction_policy = policy.validate()?;
        Ok(self)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn compaction_policy(&self) -> CompactionPolicy {
        self.compaction_policy
    }

    pub fn tombstone_count(&self) -> usize {
        self.tombstones.len()
    }

    pub fn dead_ratio(&self) -> f32 {
        dead_ratio(&self.tombstones, self.next_id as usize)
    }

    pub fn len(&self) -> usize {
        self.live_persistent_count() + self.staging_count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn live_persistent_count(&self) -> usize {
        self.records
            .keys()
            .filter(|id| !self.tombstones.contains(id))
            .count()
    }

    pub fn staging_count(&self) -> usize {
        self.staging_order.len()
    }

    pub fn has_ivf_index(&self) -> bool {
        self.ivf.is_some()
    }

    /// Append a vector to the in-memory staging tier (MCTS / ephemeral writes).
    pub fn insert_staging(
        &mut self,
        vector: &[f32],
        record: &ChunkRecord,
    ) -> Result<VectorId, DbError> {
        self.ensure_writable()?;
        self.validate_vector(vector)?;
        self.validate_record(record)?;
        if self.staging_count() >= MAX_STAGING_VECTORS {
            return Err(DbError::StagingCapacityExceeded {
                count: self.staging_count() + 1,
                max: MAX_STAGING_VECTORS,
            });
        }

        let id = self.allocate_id()?;
        self.staging_vectors.extend_from_slice(vector);
        self.staging_order.push(id);
        self.staging_records.insert(id, record.clone());

        Ok(id)
    }

    /// Legacy staging insert using a `file_path:start:end` metadata key.
    pub fn insert_staging_metadata(
        &mut self,
        vector: &[f32],
        metadata_id: &str,
    ) -> Result<VectorId, DbError> {
        self.insert_staging(vector, &record_from_metadata_id(metadata_id))
    }

    /// Append a vector directly to the persistent (append-only) tier.
    pub fn insert_persistent(
        &mut self,
        vector: &[f32],
        record: &ChunkRecord,
    ) -> Result<VectorId, DbError> {
        self.insert_persistent_inner(vector, record, true)
    }

    fn insert_persistent_inner(
        &mut self,
        vector: &[f32],
        record: &ChunkRecord,
        persist: bool,
    ) -> Result<VectorId, DbError> {
        self.ensure_writable()?;
        self.validate_vector(vector)?;
        self.validate_record(record)?;
        self.ensure_vector_capacity(vector)?;

        let id = self.allocate_id()?;
        let byte_offset = self.mmap.len() as u64;

        let vectors_path = self.dir.join(VECTORS_FILE);
        let mut file = OpenOptions::new()
            .append(true)
            .open(&vectors_path)
            .map_err(|source| DbError::io_path(&vectors_path, source))?;
        file.write_all(f32_slice_as_bytes(vector))
            .map_err(|source| DbError::io_path(&vectors_path, source))?;
        file.sync_all()
            .map_err(|source| DbError::io_path(&vectors_path, source))?;

        self.id_offsets.insert(id, byte_offset);
        self.records.insert(id, record.clone());
        self.postings.insert(id, record);
        self.assign_ivf_bucket(id, vector)?;
        self.mmap = map_vectors_file(&vectors_path)?;
        if persist {
            self.persist_manifest()?;
        }
        Ok(id)
    }

    /// Tombstone existing file chunks and append replacements (no full rewrite).
    pub fn replace_file(
        &mut self,
        file_path: &str,
        chunks: &[(&[f32], &ChunkRecord)],
    ) -> Result<(), DbError> {
        self.ensure_writable()?;
        self.delete_by_file_path(file_path)?;
        for (vector, record) in chunks {
            if record.file_path != file_path {
                return Err(DbError::FilePathMismatch {
                    expected: file_path.to_owned(),
                    got: record.file_path.clone(),
                });
            }
            self.insert_persistent_inner(vector, record, false)?;
        }
        self.persist_manifest()?;
        self.maybe_compact()?;
        Ok(())
    }

    pub fn clear_staging(&mut self) -> Result<(), DbError> {
        self.ensure_writable()?;
        self.staging_vectors.clear();
        self.staging_records.clear();
        self.staging_order.clear();
        Ok(())
    }

    pub fn flush_to_disk(&mut self) -> Result<(), DbError> {
        self.ensure_writable()?;
        if self.staging_order.is_empty() {
            return Ok(());
        }

        self.ensure_staging_flush_capacity(self.staging_order.len())?;

        let vectors_path = self.dir.join(VECTORS_FILE);
        let mut file = OpenOptions::new()
            .append(true)
            .open(&vectors_path)
            .map_err(|source| DbError::io_path(&vectors_path, source))?;

        let dim = self.dim;
        let mut file_len = self.mmap.len() as u64;
        let staging_ids = self.staging_order.clone();

        for (chunk_index, id) in staging_ids.into_iter().enumerate() {
            let start = chunk_index * dim;
            let vector = self.staging_vectors[start..start + dim].to_vec();

            file.write_all(f32_slice_as_bytes(&vector))
                .map_err(|source| DbError::io_path(&vectors_path, source))?;

            let record = self
                .staging_records
                .get(&id)
                .cloned()
                .ok_or(DbError::MissingMetadata { id })?;

            self.id_offsets.insert(id, file_len);
            file_len += (dim * std::mem::size_of::<f32>()) as u64;
            self.records.insert(id, record.clone());
            self.postings.insert(id, &record);
            self.assign_ivf_bucket(id, &vector)?;
        }

        file.sync_all()
            .map_err(|source| DbError::io_path(&vectors_path, source))?;

        self.mmap = map_vectors_file(&vectors_path)?;
        self.clear_staging()?;
        self.persist_manifest()?;
        self.maybe_compact()?;
        Ok(())
    }

    /// Tombstone all vectors for `file_path` via posting list (O(chunks in file)).
    pub fn delete_by_file_path(&mut self, file_path: &str) -> Result<usize, DbError> {
        self.ensure_writable()?;
        let mut removed = 0usize;
        let mut dirty = false;

        if let Some(ids) = self.postings.by_file.get(file_path).cloned() {
            for id in ids {
                if self.tombstone_id_inner(id)? {
                    removed += 1;
                    dirty = true;
                }
            }
        }

        let staging_drop: Vec<VectorId> = self
            .staging_records
            .iter()
            .filter(|(_, record)| record.file_path == file_path)
            .map(|(id, _)| *id)
            .collect();
        for id in staging_drop {
            self.remove_staging_id(id);
            removed += 1;
        }

        if dirty {
            self.persist_manifest()?;
        }
        self.maybe_compact()?;
        Ok(removed)
    }

    /// Rewrite `vectors.bin`, clearing tombstones and rebuilding offsets.
    pub fn compact(&mut self) -> Result<CompactionStats, DbError> {
        self.ensure_writable()?;
        let vectors = self.persistent_vectors()?.to_vec();
        let (new_vectors, new_offsets, stats) = compact_vectors(
            self.dim,
            &self.records,
            &self.id_offsets,
            &self.tombstones,
            &vectors,
        )?;

        let vectors_path = self.dir.join(VECTORS_FILE);
        // Windows refuses to rewrite a file that still has an active
        // memory-mapped section open, so drop the current mapping onto an
        // anonymous placeholder before truncating/rewriting vectors.bin.
        self.mmap = placeholder_mmap()?;
        write_vector_file(&vectors_path, &new_vectors)?;

        let tombstoned: Vec<VectorId> = self.tombstones.iter().copied().collect();
        for id in tombstoned {
            self.records.remove(&id);
        }
        self.tombstones.clear();
        self.id_offsets = new_offsets;
        self.persist_manifest()?;
        self.mmap = map_vectors_file(&vectors_path)?;
        let (num_buckets, probe_buckets) = self
            .ivf
            .as_ref()
            .map(|index| (index.num_buckets, index.probe_buckets))
            .unwrap_or((default_ivf_buckets(), default_ivf_probe()));
        self.rebuild_ivf_index(num_buckets, probe_buckets)?;
        Ok(stats)
    }

    /// Lazy compaction when dead ratio crosses the configured threshold.
    pub fn maybe_compact(&mut self) -> Result<Option<CompactionStats>, DbError> {
        if self.dead_ratio() >= self.compaction_policy.dead_ratio_threshold
            && !self.tombstones.is_empty()
        {
            Ok(Some(self.compact()?))
        } else {
            Ok(None)
        }
    }

    /// Build or rebuild the Tier-2 IVF-lite index over live persistent vectors.
    pub fn build_ivf_index(
        &mut self,
        num_buckets: usize,
        probe_buckets: usize,
    ) -> Result<(), DbError> {
        self.ensure_writable()?;
        self.rebuild_ivf_index(num_buckets, probe_buckets)
    }

    pub fn reset(&mut self) -> Result<(), DbError> {
        self.ensure_writable()?;
        let vectors_path = self.dir.join(VECTORS_FILE);
        let manifest_path = self.dir.join(MANIFEST_FILE);
        let centroids = centroids_path(&self.dir);

        // Same Windows constraint as `compact()`: drop the mapped section
        // before truncating vectors.bin.
        self.mmap = placeholder_mmap()?;
        File::create(&vectors_path).map_err(|source| DbError::io_path(&vectors_path, source))?;
        let manifest = Manifest::default();
        write_manifest(&manifest_path, &manifest)?;
        let _ = fs::remove_file(centroids);

        self.dim = manifest.dim;
        self.next_id = 0;
        self.records.clear();
        self.id_offsets.clear();
        self.tombstones.clear();
        self.postings = PostingIndex::default();
        self.ivf = None;
        self.compaction_policy = manifest.compaction;
        self.clear_staging()?;
        self.mmap = map_vectors_file(&vectors_path)?;
        Ok(())
    }

    pub(crate) fn persistent_vectors(&self) -> Result<&[f32], DbError> {
        f32_slice_from_mmap(&self.mmap)
    }

    pub(crate) fn staging_order(&self) -> &[VectorId] {
        &self.staging_order
    }

    pub(crate) fn record_for_id(&self, id: VectorId) -> Option<&ChunkRecord> {
        self.staging_records
            .get(&id)
            .or_else(|| self.records.get(&id))
    }

    pub(crate) fn is_live(&self, id: VectorId) -> bool {
        !self.tombstones.contains(&id)
    }

    pub(crate) fn tombstones(&self) -> &HashSet<VectorId> {
        &self.tombstones
    }

    pub(crate) fn postings(&self) -> &PostingIndex {
        &self.postings
    }

    pub(crate) fn ivf(&self) -> Option<&IvfIndex> {
        self.ivf.as_ref()
    }

    pub(crate) fn live_persistent_ids(&self) -> Vec<VectorId> {
        self.records
            .keys()
            .copied()
            .filter(|id| self.is_live(*id))
            .collect()
    }

    pub(crate) fn persistent_vector_at(&self, id: VectorId) -> Option<&[f32]> {
        if !self.is_live(id) {
            return None;
        }
        let byte_offset = self.id_offsets.get(&id)?;
        let start = (*byte_offset as usize) / std::mem::size_of::<f32>();
        let end = start + self.dim;
        self.persistent_vectors().ok()?.get(start..end)
    }

    pub(crate) fn staging_vector_at(&self, id: VectorId) -> Option<&[f32]> {
        let index = self
            .staging_order
            .iter()
            .position(|existing| *existing == id)?;
        let start = index * self.dim;
        let end = start + self.dim;
        self.staging_vectors.get(start..end)
    }

    fn tombstone_id_inner(&mut self, id: VectorId) -> Result<bool, DbError> {
        if self.tombstones.contains(&id) {
            return Ok(false);
        }
        let record = self
            .records
            .get(&id)
            .cloned()
            .ok_or(DbError::MissingMetadata { id })?;
        self.tombstones.insert(id);
        self.postings.remove(id, &record);
        if let Some(ivf) = self.ivf.as_mut() {
            ivf.assignments.remove(&id);
        }
        Ok(true)
    }

    fn allocate_id(&mut self) -> Result<VectorId, DbError> {
        if self.next_id == VectorId::MAX {
            return Err(DbError::IdExhausted);
        }
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        Ok(id)
    }

    fn remove_staging_id(&mut self, id: VectorId) {
        let Some(index) = self
            .staging_order
            .iter()
            .position(|existing| *existing == id)
        else {
            return;
        };

        let dim = self.dim;
        let start = index * dim;
        let end = start + dim;
        self.staging_vectors.drain(start..end);
        self.staging_order.remove(index);
        self.staging_records.remove(&id);
    }

    fn assign_ivf_bucket(&mut self, id: VectorId, vector: &[f32]) -> Result<(), DbError> {
        let Some(ivf) = self.ivf.as_mut() else {
            return Ok(());
        };
        let bucket = ivf.nearest_bucket(vector, self.dim);
        ivf.assignments.insert(id, bucket);
        Ok(())
    }

    fn rebuild_ivf_index(
        &mut self,
        num_buckets: usize,
        probe_buckets: usize,
    ) -> Result<(), DbError> {
        self.ensure_writable()?;
        let live_ids = self.live_persistent_ids();
        if live_ids.len() < num_buckets {
            self.ivf = None;
            let _ = fs::remove_file(centroids_path(&self.dir));
            self.persist_manifest()?;
            return Ok(());
        }

        let index = build_ivf_index(
            num_buckets,
            probe_buckets,
            self.dim,
            &live_ids,
            self.persistent_vectors()?,
            &self.id_offsets,
        )?;

        write_centroids(&centroids_path(&self.dir), &index.centroids)?;
        self.ivf = Some(index);
        self.persist_manifest()?;
        Ok(())
    }

    fn validate_record(&self, record: &ChunkRecord) -> Result<(), DbError> {
        if record.text.len() > MAX_RECORD_TEXT_BYTES {
            return Err(DbError::RecordTextTooLarge {
                len: record.text.len(),
                max: MAX_RECORD_TEXT_BYTES,
            });
        }
        Ok(())
    }

    fn ensure_vector_capacity(&self, vector: &[f32]) -> Result<(), DbError> {
        let additional = std::mem::size_of_val(vector) as u64;
        let projected = self.mmap.len() as u64 + additional;
        if projected > MAX_VECTORS_BYTES {
            return Err(DbError::VectorsFileTooLarge {
                len: projected,
                max: MAX_VECTORS_BYTES,
            });
        }
        Ok(())
    }

    fn ensure_staging_flush_capacity(&self, row_count: usize) -> Result<(), DbError> {
        let additional = (row_count * self.dim * std::mem::size_of::<f32>()) as u64;
        let projected = self.mmap.len() as u64 + additional;
        if projected > MAX_VECTORS_BYTES {
            return Err(DbError::VectorsFileTooLarge {
                len: projected,
                max: MAX_VECTORS_BYTES,
            });
        }
        Ok(())
    }

    /// Tombstone a single vector id.
    pub fn tombstone(&mut self, id: VectorId) -> Result<bool, DbError> {
        self.ensure_writable()?;
        let dirty = self.tombstone_id_inner(id)?;
        if dirty {
            self.persist_manifest()?;
            self.maybe_compact()?;
        }
        Ok(dirty)
    }

    /// Fetch a live vector and its metadata by id.
    pub fn get(&self, id: VectorId) -> Result<Point, DbError> {
        let record = self
            .record_for_id(id)
            .cloned()
            .ok_or(DbError::MissingMetadata { id })?;

        let vector = if self.staging_records.contains_key(&id) {
            self.staging_vector_at(id)
                .map(|slice| slice.to_vec())
                .ok_or(DbError::MissingOffset { id })
        } else if self.is_live(id) {
            self.persistent_vector_at(id)
                .map(|slice| slice.to_vec())
                .ok_or(DbError::MissingOffset { id })
        } else {
            return Err(DbError::MissingMetadata { id });
        }?;

        Ok(Point { id, vector, record })
    }

    fn validate_vector(&self, vector: &[f32]) -> Result<(), DbError> {
        if vector.len() != self.dim {
            return Err(DbError::InvalidDimension {
                expected: self.dim,
                got: vector.len(),
            });
        }
        Ok(())
    }

    fn persist_manifest(&self) -> Result<(), DbError> {
        let manifest_path = self.dir.join(MANIFEST_FILE);
        let ivf = self.ivf.as_ref().map(|index| IvfManifest {
            num_buckets: index.num_buckets,
            probe_buckets: index.probe_buckets,
            assignments: index
                .assignments
                .iter()
                .map(|(id, bucket)| (id.to_string(), *bucket))
                .collect(),
        });

        let manifest = Manifest {
            version: MANIFEST_VERSION,
            dim: self.dim,
            next_id: self.next_id,
            compaction: self.compaction_policy,
            records: encode_record_map(&self.records),
            offsets: encode_offset_map(&self.id_offsets),
            tombstones: self.tombstones.iter().copied().collect(),
            postings: self.postings.clone(),
            ivf,
        };
        write_manifest(&manifest_path, &manifest)
    }
}

fn validate_open_state(
    dim: usize,
    records: &HashMap<VectorId, ChunkRecord>,
    id_offsets: &HashMap<VectorId, u64>,
    tombstones: &HashSet<VectorId>,
    mmap_len: usize,
) -> Result<(), DbError> {
    if dim == 0 {
        return Err(DbError::InvalidDimension {
            expected: 1,
            got: 0,
        });
    }

    if records.keys().copied().collect::<HashSet<_>>() != id_offsets.keys().copied().collect() {
        return Err(DbError::CorruptManifestKeyMismatch);
    }

    for &id in tombstones {
        if !records.contains_key(&id) {
            return Err(DbError::CorruptManifestTombstone { id });
        }
    }

    let vector_bytes = dim.saturating_mul(std::mem::size_of::<f32>());
    let mut seen_offsets: HashMap<u64, VectorId> = HashMap::new();
    let mut required_len = 0usize;

    for (&id, offset) in id_offsets {
        if let Some(first) = seen_offsets.insert(*offset, id) {
            return Err(DbError::DuplicateVectorOffset {
                offset: *offset,
                first,
                second: id,
            });
        }

        let end = *offset as usize + vector_bytes;
        if end > mmap_len {
            return Err(DbError::CorruptVectorOffset { id });
        }
        required_len = required_len.max(end);
    }

    if required_len > mmap_len {
        return Err(DbError::VectorsFileTooSmall {
            file_len: mmap_len,
            required: required_len,
        });
    }

    Ok(())
}

fn record_from_metadata_id(metadata_id: &str) -> ChunkRecord {
    // Format is "{file_path}:{start_line}:{end_line}", but `file_path` itself
    // may legitimately contain ':' (e.g. Windows drive letters like `C:\...`).
    // Split from the right so at most the trailing two ':'-delimited fields
    // are treated as line numbers; everything else stays part of the path.
    let parts: Vec<&str> = metadata_id.rsplitn(3, ':').collect();
    let (file_path, start_line, end_line) = match parts.as_slice() {
        [end, start, path] => (
            (*path).to_owned(),
            start.parse().unwrap_or(0),
            end.parse().unwrap_or(0),
        ),
        [start, path] => {
            let start_line: u64 = start.parse().unwrap_or(0);
            ((*path).to_owned(), start_line, start_line)
        }
        _ => (metadata_id.to_owned(), 0, 0),
    };

    ChunkRecord {
        text: String::new(),
        kind: "Unknown".into(),
        name: None,
        file_path,
        start_line,
        end_line,
    }
}

fn load_ivf(
    dir: &Path,
    dim: usize,
    manifest: Option<IvfManifest>,
) -> Result<Option<IvfIndex>, DbError> {
    let Some(meta) = manifest else {
        return Ok(None);
    };

    // If centroids.bin is missing or unreadable (deleted, truncated, dim
    // mismatch after a manual edit, etc.) degrade gracefully to "no IVF
    // index" instead of failing the entire `open()`. Callers can rebuild it
    // via `build_ivf_index()`.
    let centroids = read_centroids(&centroids_path(dir), meta.num_buckets, dim).map_err(|err| {
        DbError::CorruptCentroids {
            reason: err.to_string(),
        }
    })?;
    let assignments = meta
        .assignments
        .into_iter()
        .filter_map(|(id, bucket)| id.parse::<VectorId>().ok().map(|id| (id, bucket)))
        .collect();

    Ok(Some(IvfIndex {
        num_buckets: meta.num_buckets,
        probe_buckets: meta.probe_buckets,
        assignments,
        centroids,
    }))
}

fn write_manifest(path: &Path, manifest: &Manifest) -> Result<(), DbError> {
    let json = serde_json::to_vec(manifest)?;
    if json.len() > MAX_MANIFEST_BYTES {
        return Err(DbError::ManifestTooLarge {
            len: json.len(),
            max: MAX_MANIFEST_BYTES,
        });
    }

    // Write to a temp file and rename into place so a crash/power-loss mid-write
    // can never leave a truncated/corrupt manifest.json behind.
    let mut tmp_name = path.as_os_str().to_owned();
    tmp_name.push(".tmp");
    let tmp_path = PathBuf::from(tmp_name);

    let mut file = File::create(&tmp_path).map_err(|source| DbError::io_path(&tmp_path, source))?;
    file.write_all(&json)
        .map_err(|source| DbError::io_path(&tmp_path, source))?;
    file.sync_all()
        .map_err(|source| DbError::io_path(&tmp_path, source))?;
    drop(file);

    fs::rename(&tmp_path, path).map_err(|source| DbError::io_path(path, source))
}

fn write_vector_file(path: &Path, vectors: &[f32]) -> Result<(), DbError> {
    let tmp_path = path.with_extension("bin.tmp");
    let mut file = File::create(&tmp_path).map_err(|source| DbError::io_path(&tmp_path, source))?;
    file.write_all(f32_slice_as_bytes(vectors))
        .map_err(|source| DbError::io_path(&tmp_path, source))?;
    file.sync_all()
        .map_err(|source| DbError::io_path(&tmp_path, source))?;
    drop(file);
    fs::rename(&tmp_path, path).map_err(|source| DbError::io_path(path, source))
}

fn map_vectors_file(path: &Path) -> Result<Mmap, DbError> {
    let file = File::open(path).map_err(|source| DbError::io_path(path, source))?;
    unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|error| DbError::Mmap(error.to_string()))
    }
}

/// An anonymous (not file-backed) empty mapping used to release the OS-level
/// reference to `vectors.bin` before it's rewritten in place. Needed because
/// Windows refuses to truncate/replace a file with an active mapped section.
fn placeholder_mmap() -> Result<Mmap, DbError> {
    MmapOptions::new()
        .len(1)
        .map_anon()
        .and_then(MmapMut::make_read_only)
        .map_err(|error| DbError::Mmap(error.to_string()))
}

/// Reinterprets `mmap`'s bytes as `f32`s with zero copy.
///
/// # Portability
///
/// This assumes the host's native endianness matches the endianness the
/// bytes were written with (see [`f32_slice_as_bytes`]), so `vectors.bin` is
/// only portable between hosts that share the same endianness (e.g.
/// x86_64/ARM, both little-endian). Reading a file written on a big-endian
/// host would silently misinterpret the data. This tradeoff is intentional:
/// it keeps reads zero-copy off the mmap.
pub(crate) fn f32_slice_from_mmap(mmap: &Mmap) -> Result<&[f32], DbError> {
    let bytes = mmap.as_ref();
    if bytes.is_empty() {
        return Ok(&[]);
    }
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(DbError::InvalidVectorBytes { len: bytes.len() });
    }

    let (prefix, aligned, suffix) = unsafe { bytes.align_to::<f32>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(DbError::UnalignedVectorData { len: bytes.len() });
    }

    Ok(aligned)
}

/// Reinterprets a `f32` slice as raw bytes with zero copy, using the host's
/// native endianness (see [`f32_slice_from_mmap`] for the portability caveat
/// this implies).
fn f32_slice_as_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn encode_record_map(map: &HashMap<VectorId, ChunkRecord>) -> HashMap<String, ChunkRecord> {
    map.iter()
        .map(|(id, record)| (id.to_string(), record.clone()))
        .collect()
}

fn decode_record_map(
    entries: &HashMap<String, ChunkRecord>,
) -> Result<HashMap<VectorId, ChunkRecord>, DbError> {
    entries
        .iter()
        .map(|(id, record)| {
            id.parse::<VectorId>()
                .map(|id| (id, record.clone()))
                .map_err(|_| DbError::InvalidManifestId { value: id.clone() })
        })
        .collect()
}

fn encode_offset_map(map: &HashMap<VectorId, u64>) -> HashMap<String, u64> {
    map.iter()
        .map(|(id, offset)| (id.to_string(), *offset))
        .collect()
}

fn decode_offset_map(entries: &HashMap<String, u64>) -> Result<HashMap<VectorId, u64>, DbError> {
    entries
        .iter()
        .map(|(id, offset)| {
            id.parse::<VectorId>()
                .map(|id| (id, *offset))
                .map_err(|_| DbError::InvalidManifestId { value: id.clone() })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::dot_product;

    #[test]
    fn open_creates_empty_database() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Rek0nDb::open(dir.path()).expect("open");
        assert!(db.is_empty());
    }

    #[test]
    fn mmap_round_trip_preserves_dot_product() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vectors_path = dir.path().join(VECTORS_FILE);
        let vector = {
            let mut values = vec![0.0_f32; EMBEDDING_DIM];
            values[7] = 1.0;
            values
        };
        write_vector_file(&vectors_path, &vector).expect("write");

        let mmap = map_vectors_file(&vectors_path).expect("mmap");
        let slice = f32_slice_from_mmap(&mmap).expect("cast");
        assert_eq!(slice.len(), EMBEDDING_DIM);
        assert!((dot_product(slice, &vector) - 1.0).abs() < f32::EPSILON);
    }
}
