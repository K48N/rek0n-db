use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapOptions};
use serde::{Deserialize, Serialize};

use crate::compact::{compact_vectors, dead_ratio};
use crate::ivf::{
    build_ivf_index, centroids_path, default_ivf_buckets, default_ivf_probe, read_centroids,
    write_centroids, IvfIndex,
};
use crate::postings::PostingIndex;
use crate::types::{
    ChunkRecord, CompactionPolicy, CompactionStats, DbError, VectorId, EMBEDDING_DIM,
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
}

impl Rek0nDb {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, DbError> {
        let dir = dir.as_ref().to_path_buf();
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
            serde_json::from_slice(&bytes)?
        } else {
            let default = Manifest::default();
            write_manifest(&manifest_path, &default)?;
            default
        };

        let mmap = map_vectors_file(&vectors_path)?;
        let records = decode_record_map(&manifest.records)?;
        let id_offsets = decode_offset_map(&manifest.offsets)?;
        let tombstones = manifest.tombstones.into_iter().collect();
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
            compaction_policy: manifest.compaction,
            staging_vectors: Vec::new(),
            staging_records: HashMap::new(),
            staging_order: Vec::new(),
        })
    }

    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    pub fn with_compaction_policy(mut self, policy: CompactionPolicy) -> Self {
        self.compaction_policy = policy;
        self
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
        self.validate_vector(vector)?;

        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);

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
        self.validate_vector(vector)?;

        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
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
        self.persist_manifest()?;
        Ok(id)
    }

    /// Tombstone existing file chunks and append replacements (no full rewrite).
    pub fn replace_file(
        &mut self,
        file_path: &str,
        chunks: &[(&[f32], &ChunkRecord)],
    ) -> Result<(), DbError> {
        self.delete_by_file_path(file_path)?;
        for (vector, record) in chunks {
            if record.file_path != file_path {
                return Err(DbError::InvalidQuery {
                    expected: 0,
                    got: 0,
                });
            }
            self.insert_persistent(vector, record)?;
        }
        self.maybe_compact()?;
        Ok(())
    }

    pub fn clear_staging(&mut self) {
        self.staging_vectors.clear();
        self.staging_records.clear();
        self.staging_order.clear();
    }

    pub fn flush_to_disk(&mut self) -> Result<(), DbError> {
        if self.staging_order.is_empty() {
            return Ok(());
        }

        let vectors_path = self.dir.join(VECTORS_FILE);
        let mut file = OpenOptions::new()
            .append(true)
            .open(&vectors_path)
            .map_err(|source| DbError::io_path(&vectors_path, source))?;

        let dim = self.dim;
        let mut file_len = self.mmap.len() as u64;
        let staging_ids = self.staging_order.clone();

        for id in staging_ids {
            let chunk_index = self
                .staging_order
                .iter()
                .position(|existing| *existing == id)
                .expect("staging order");
            let start = chunk_index * dim;
            let vector = self.staging_vectors[start..start + dim].to_vec();

            file.write_all(f32_slice_as_bytes(&vector))
                .map_err(|source| DbError::io_path(&vectors_path, source))?;

            let record = self
                .staging_records
                .get(&id)
                .cloned()
                .expect("staging record");

            self.id_offsets.insert(id, file_len);
            file_len += (dim * std::mem::size_of::<f32>()) as u64;
            self.records.insert(id, record.clone());
            self.postings.insert(id, &record);
            self.assign_ivf_bucket(id, &vector)?;
        }

        file.sync_all()
            .map_err(|source| DbError::io_path(&vectors_path, source))?;

        self.mmap = map_vectors_file(&vectors_path)?;
        self.clear_staging();
        self.persist_manifest()?;
        self.maybe_compact()?;
        Ok(())
    }

    /// Tombstone all vectors for `file_path` via posting list (O(chunks in file)).
    pub fn delete_by_file_path(&mut self, file_path: &str) -> Result<usize, DbError> {
        let mut removed = 0usize;

        if let Some(ids) = self.postings.by_file.get(file_path).cloned() {
            for id in ids {
                self.tombstone_id(id)?;
                removed += 1;
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

        self.maybe_compact()?;
        Ok(removed)
    }

    /// Rewrite `vectors.bin`, clearing tombstones and rebuilding offsets.
    pub fn compact(&mut self) -> Result<CompactionStats, DbError> {
        let vectors = self.persistent_vectors().to_vec();
        let (new_vectors, new_offsets, stats) = compact_vectors(
            self.dim,
            &self.records,
            &self.id_offsets,
            &self.tombstones,
            &vectors,
        )?;

        let vectors_path = self.dir.join(VECTORS_FILE);
        write_vector_file(&vectors_path, &new_vectors)?;

        let tombstoned: Vec<VectorId> = self.tombstones.iter().copied().collect();
        for id in tombstoned {
            self.records.remove(&id);
        }
        self.tombstones.clear();
        self.id_offsets = new_offsets;
        self.mmap = map_vectors_file(&vectors_path)?;
        self.rebuild_ivf_index(default_ivf_buckets(), default_ivf_probe())?;
        self.persist_manifest()?;
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
        self.rebuild_ivf_index(num_buckets, probe_buckets)
    }

    pub fn reset(&mut self) -> Result<(), DbError> {
        let vectors_path = self.dir.join(VECTORS_FILE);
        let manifest_path = self.dir.join(MANIFEST_FILE);
        let centroids = centroids_path(&self.dir);

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
        self.clear_staging();
        self.mmap = map_vectors_file(&vectors_path)?;
        Ok(())
    }

    pub(crate) fn persistent_vectors(&self) -> &[f32] {
        f32_slice_from_mmap(&self.mmap).unwrap_or(&[])
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
        self.persistent_vectors().get(start..end)
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

    fn tombstone_id(&mut self, id: VectorId) -> Result<(), DbError> {
        if self.tombstones.contains(&id) {
            return Ok(());
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
        self.persist_manifest()?;
        Ok(())
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
            self.persistent_vectors(),
            &self.id_offsets,
        )?;

        write_centroids(&centroids_path(&self.dir), &index.centroids)?;
        self.ivf = Some(index);
        self.persist_manifest()?;
        Ok(())
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

fn record_from_metadata_id(metadata_id: &str) -> ChunkRecord {
    let mut parts = metadata_id.splitn(3, ':');
    let file_path = parts.next().unwrap_or("unknown").to_owned();
    let start_line = parts
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let end_line = parts
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(start_line);

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
    let centroids = read_centroids(&centroids_path(dir), meta.num_buckets, dim)?;
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
    let json = serde_json::to_vec_pretty(manifest)?;
    fs::write(path, json).map_err(|source| DbError::io_path(path, source))
}

fn write_vector_file(path: &Path, vectors: &[f32]) -> Result<(), DbError> {
    fs::write(path, f32_slice_as_bytes(vectors)).map_err(|source| DbError::io_path(path, source))
}

fn map_vectors_file(path: &Path) -> Result<Mmap, DbError> {
    let file = File::open(path).map_err(|source| DbError::io_path(path, source))?;
    unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|error| DbError::Mmap(error.to_string()))
    }
}

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
                .map_err(|_| DbError::InvalidQuery {
                    expected: 0,
                    got: 0,
                })
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
                .map_err(|_| DbError::InvalidQuery {
                    expected: 0,
                    got: 0,
                })
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
