use std::collections::{HashMap, HashSet};

use crate::ivf::read_vector;
use crate::types::{CompactionStats, DbError, VectorId};

pub type CompactResult = (Vec<f32>, HashMap<VectorId, u64>, CompactionStats);

/// Rewrite `vectors.bin`, dropping tombstoned rows and rebuilding offsets.
pub fn compact_vectors(
    dim: usize,
    records: &HashMap<VectorId, crate::types::ChunkRecord>,
    id_offsets: &HashMap<VectorId, u64>,
    tombstones: &HashSet<VectorId>,
    old_vectors: &[f32],
) -> Result<CompactResult, DbError> {
    let vectors_before = records.len();
    let bytes_before = old_vectors.len() as u64 * std::mem::size_of::<f32>() as u64;

    let mut live_ids: Vec<VectorId> = records
        .keys()
        .copied()
        .filter(|id| !tombstones.contains(id))
        .collect();
    live_ids.sort_unstable();

    let mut new_vectors = Vec::with_capacity(live_ids.len() * dim);
    let mut new_offsets = HashMap::with_capacity(live_ids.len());

    for id in &live_ids {
        let vector = read_vector(old_vectors, dim, id_offsets, *id)?;
        let byte_offset = (new_vectors.len() * std::mem::size_of::<f32>()) as u64;
        new_offsets.insert(*id, byte_offset);
        new_vectors.extend_from_slice(vector);
    }

    let bytes_after = new_vectors.len() as u64 * std::mem::size_of::<f32>() as u64;

    Ok((
        new_vectors,
        new_offsets,
        CompactionStats {
            vectors_before,
            vectors_after: live_ids.len(),
            bytes_reclaimed: bytes_before.saturating_sub(bytes_after),
        },
    ))
}

pub fn dead_ratio(tombstones: &HashSet<VectorId>, allocated_ids: usize) -> f32 {
    if allocated_ids == 0 {
        0.0
    } else {
        tombstones.len() as f32 / allocated_ids as f32
    }
}
