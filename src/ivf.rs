use std::collections::{HashMap, HashSet};

use crate::search::dot_product;
use crate::types::{DbError, VectorId, DEFAULT_IVF_BUCKETS, DEFAULT_IVF_PROBE};

const CENTROIDS_FILE: &str = "centroids.bin";

/// IVF-lite index: flat centroid table + per-vector bucket assignments.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct IvfIndex {
    pub num_buckets: usize,
    pub probe_buckets: usize,
    pub assignments: HashMap<VectorId, usize>,
    #[serde(skip)]
    pub centroids: Vec<f32>,
}

impl IvfIndex {
    pub fn bucket_ids(&self, bucket: usize) -> impl Iterator<Item = VectorId> + '_ {
        self.assignments
            .iter()
            .filter_map(move |(id, assigned)| (*assigned == bucket).then_some(*id))
    }

    /// Pick the `probe_buckets` centroids closest to the query.
    pub fn probe(&self, query: &[f32], dim: usize, probe_buckets: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = (0..self.num_buckets)
            .map(|bucket| {
                let start = bucket * dim;
                let centroid = &self.centroids[start..start + dim];
                (bucket, dot_product(query, centroid))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(probe_buckets.min(self.num_buckets))
            .map(|(bucket, _)| bucket)
            .collect()
    }

    pub fn nearest_bucket(&self, vector: &[f32], dim: usize) -> usize {
        self.probe(vector, dim, 1)[0]
    }

    /// Candidate ids from probed buckets, minus tombstones.
    pub fn candidates(
        &self,
        query: &[f32],
        dim: usize,
        probe_buckets: usize,
        tombstones: &HashSet<VectorId>,
    ) -> Vec<VectorId> {
        let mut set = HashSet::new();
        for bucket in self.probe(query, dim, probe_buckets) {
            for id in self.bucket_ids(bucket) {
                if !tombstones.contains(&id) {
                    set.insert(id);
                }
            }
        }
        set.into_iter().collect()
    }
}

/// Train IVF-lite buckets from live persistent vectors.
pub fn build_ivf_index(
    num_buckets: usize,
    probe_buckets: usize,
    dim: usize,
    live_ids: &[VectorId],
    vectors: &[f32],
    id_offsets: &HashMap<VectorId, u64>,
) -> Result<IvfIndex, DbError> {
    if num_buckets == 0 {
        return Err(DbError::InvalidIvfBucketCount);
    }
    if live_ids.len() < num_buckets {
        return Err(DbError::InsufficientVectorsForIvf {
            live: live_ids.len(),
            buckets: num_buckets,
        });
    }

    let centroids = train_centroids(num_buckets, dim, live_ids, vectors, id_offsets)?;
    let mut assignments = HashMap::with_capacity(live_ids.len());

    for &id in live_ids {
        let vector = read_vector(vectors, dim, id_offsets, id)?;
        let bucket = nearest_centroid(vector, dim, &centroids, num_buckets);
        assignments.insert(id, bucket);
    }

    Ok(IvfIndex {
        num_buckets,
        probe_buckets,
        assignments,
        centroids,
    })
}

pub fn centroids_path(dir: &std::path::Path) -> std::path::PathBuf {
    dir.join(CENTROIDS_FILE)
}

pub fn write_centroids(path: &std::path::Path, centroids: &[f32]) -> Result<(), DbError> {
    use std::fs::{self, File};
    use std::io::Write;

    let bytes = unsafe {
        std::slice::from_raw_parts(
            centroids.as_ptr().cast::<u8>(),
            std::mem::size_of_val(centroids),
        )
    };

    let tmp_path = path.with_extension("bin.tmp");
    let mut file =
        File::create(&tmp_path).map_err(|source| DbError::io_path(&tmp_path, source))?;
    file.write_all(bytes)
        .map_err(|source| DbError::io_path(&tmp_path, source))?;
    file.sync_all()
        .map_err(|source| DbError::io_path(&tmp_path, source))?;
    drop(file);
    fs::rename(&tmp_path, path).map_err(|source| DbError::io_path(path, source))
}

pub fn read_centroids(
    path: &std::path::Path,
    num_buckets: usize,
    dim: usize,
) -> Result<Vec<f32>, DbError> {
    let bytes = std::fs::read(path).map_err(|source| DbError::io_path(path, source))?;
    let expected = num_buckets * dim * std::mem::size_of::<f32>();
    if bytes.len() != expected {
        return Err(DbError::InvalidVectorBytes { len: bytes.len() });
    }
    let mut centroids = vec![0.0_f32; num_buckets * dim];
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let bytes: [u8; 4] = chunk
            .try_into()
            .map_err(|_| DbError::InvalidVectorBytes { len: bytes.len() })?;
        centroids[index] = f32::from_le_bytes(bytes);
    }
    Ok(centroids)
}

pub(crate) fn read_vector<'a>(
    vectors: &'a [f32],
    dim: usize,
    id_offsets: &HashMap<VectorId, u64>,
    id: VectorId,
) -> Result<&'a [f32], DbError> {
    let byte_offset = id_offsets
        .get(&id)
        .copied()
        .ok_or(DbError::MissingOffset { id })?;
    let start = (byte_offset as usize) / std::mem::size_of::<f32>();
    let end = start + dim;
    vectors.get(start..end).ok_or(DbError::MissingOffset { id })
}

fn train_centroids(
    num_buckets: usize,
    dim: usize,
    live_ids: &[VectorId],
    vectors: &[f32],
    id_offsets: &HashMap<VectorId, u64>,
) -> Result<Vec<f32>, DbError> {
    // k-means-lite: pick evenly spaced live vectors as initial centroids, one refinement pass.
    let stride = (live_ids.len() / num_buckets).max(1);
    let mut centroids = Vec::with_capacity(num_buckets * dim);

    for bucket in 0..num_buckets {
        let pick = live_ids[(bucket * stride).min(live_ids.len() - 1)];
        let vector = read_vector(vectors, dim, id_offsets, pick)?;
        centroids.extend_from_slice(vector);
    }

    for _ in 0..2 {
        let mut sums = vec![0.0_f32; num_buckets * dim];
        let mut counts = vec![0_u32; num_buckets];

        for &id in live_ids {
            let vector = read_vector(vectors, dim, id_offsets, id)?;
            let bucket = nearest_centroid(vector, dim, &centroids, num_buckets);
            counts[bucket] += 1;
            let start = bucket * dim;
            for (index, value) in vector.iter().enumerate() {
                sums[start + index] += value;
            }
        }

        for (bucket, count) in counts.iter().enumerate() {
            if *count == 0 {
                continue;
            }
            let inv = 1.0 / *count as f32;
            let start = bucket * dim;
            for index in 0..dim {
                centroids[start + index] = sums[start + index] * inv;
            }
        }
    }

    Ok(centroids)
}

fn nearest_centroid(vector: &[f32], dim: usize, centroids: &[f32], num_buckets: usize) -> usize {
    let mut best_bucket = 0;
    let mut best_score = f32::NEG_INFINITY;

    for bucket in 0..num_buckets {
        let start = bucket * dim;
        let score = dot_product(vector, &centroids[start..start + dim]);
        if score > best_score {
            best_score = score;
            best_bucket = bucket;
        }
    }

    best_bucket
}

pub fn default_ivf_buckets() -> usize {
    DEFAULT_IVF_BUCKETS
}

pub fn default_ivf_probe() -> usize {
    DEFAULT_IVF_PROBE
}
