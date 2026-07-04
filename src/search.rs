use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::types::{AnnStrategy, ChunkRecord, DbError, SearchScope, VectorId};
use crate::Rek0nDb;

/// Dot product for L2-normalized vectors (cosine similarity).
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub score: f32,
    pub id: VectorId,
    pub record: ChunkRecord,
}

struct Candidate {
    score: f32,
    id: VectorId,
    record: ChunkRecord,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.id == other.id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl Rek0nDb {
    /// Tier 0 exact search over all live vectors (persistent + staging).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchHit>, DbError> {
        self.search_scoped(query, k, SearchScope::all(), AnnStrategy::Exact)
    }

    /// Scoped search with explicit ANN tier selection.
    ///
    /// - **Tier 0** (`AnnStrategy::Exact` + unrestricted scope): scan all live vectors.
    /// - **Tier 1** (`AnnStrategy::Exact` + `SearchScope` filters): exact search on posting lists.
    /// - **Tier 2** (`AnnStrategy::Ivf`): probe IVF buckets, exact search within union.
    /// - **Tier 3** (`AnnStrategy::Hnsw`): reserved until `rek0n-search` exists.
    pub fn search_scoped(
        &self,
        query: &[f32],
        k: usize,
        scope: SearchScope<'_>,
        strategy: AnnStrategy,
    ) -> Result<Vec<SearchHit>, DbError> {
        if query.len() != self.dim() {
            return Err(DbError::InvalidQuery {
                expected: self.dim(),
                got: query.len(),
            });
        }
        if k == 0 {
            return Err(DbError::InvalidSearchLimit);
        }

        self.persistent_vectors()?;

        let candidates = self.resolve_candidates(query, &scope, strategy)?;
        let mut heap = BinaryHeap::with_capacity(k.min(self.len().max(1)));

        match candidates {
            CandidateSet::All => {
                for id in self.live_persistent_ids() {
                    self.score_id(id, query, k, &mut heap, false);
                }
                if scope.include_staging {
                    for &id in self.staging_order() {
                        self.score_id(id, query, k, &mut heap, true);
                    }
                }
            }
            CandidateSet::Ids(ids) => {
                for id in ids {
                    let staging = self.staging_vector_at(id).is_some();
                    if staging && !scope.include_staging {
                        continue;
                    }
                    self.score_id(id, query, k, &mut heap, staging);
                }
            }
        }

        let mut hits: Vec<SearchHit> = heap
            .into_iter()
            .map(|candidate| SearchHit {
                score: candidate.score,
                id: candidate.id,
                record: candidate.record,
            })
            .collect();
        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(hits)
    }

    fn resolve_candidates(
        &self,
        query: &[f32],
        scope: &SearchScope<'_>,
        strategy: AnnStrategy,
    ) -> Result<CandidateSet, DbError> {
        let scoped = self.postings().resolve(
            scope.file_paths,
            scope.file_path_prefix,
            scope.kinds,
            scope.candidate_ids,
            self.tombstones(),
        );

        match strategy {
            AnnStrategy::Exact => {
                if let Some(ids) = scoped {
                    Ok(CandidateSet::Ids(ids))
                } else {
                    Ok(CandidateSet::All)
                }
            }
            AnnStrategy::Ivf { probe_buckets } => {
                let ivf = self.ivf().ok_or(DbError::IvfNotBuilt)?;
                let probe = probe_buckets.max(1);
                let mut ivf_ids: HashSet<VectorId> = ivf
                    .candidates(query, self.dim(), probe, self.tombstones())
                    .into_iter()
                    .collect();

                if scope.include_staging {
                    for &id in self.staging_order() {
                        ivf_ids.insert(id);
                    }
                }

                if let Some(mut scoped_ids) = scoped {
                    scoped_ids.retain(|id| ivf_ids.contains(id));
                    Ok(CandidateSet::Ids(scoped_ids))
                } else {
                    Ok(CandidateSet::Ids(ivf_ids.into_iter().collect()))
                }
            }
            AnnStrategy::Hnsw { ef_search: _ } => Err(DbError::HnswNotBuilt),
        }
    }

    fn score_id(
        &self,
        id: VectorId,
        query: &[f32],
        k: usize,
        heap: &mut BinaryHeap<Candidate>,
        staging: bool,
    ) {
        if !self.is_live(id) && !staging {
            return;
        }

        let vector = if staging {
            self.staging_vector_at(id)
        } else {
            self.persistent_vector_at(id)
        };
        let Some(vector) = vector else {
            return;
        };

        let record = match self.record_for_id(id).cloned() {
            Some(record) => record,
            None => return,
        };

        let score = dot_product(query, vector);
        push_candidate(heap, k, Candidate { score, id, record });
    }
}

enum CandidateSet {
    All,
    Ids(Vec<VectorId>),
}

fn push_candidate(heap: &mut BinaryHeap<Candidate>, k: usize, candidate: Candidate) {
    if heap.len() < k {
        heap.push(candidate);
    } else if let Some(weakest) = heap.peek() {
        if candidate.score > weakest.score {
            heap.pop();
            heap.push(candidate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_matches_manual() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.5_f32, 0.5, 0.0];
        assert!((dot_product(&a, &b) - 0.5).abs() < f32::EPSILON);
    }
}
