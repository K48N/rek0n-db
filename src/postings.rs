use std::collections::{HashMap, HashSet};

use crate::types::{ChunkRecord, VectorId};

/// Inverted posting lists for scoped search (Tier 1).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PostingIndex {
    pub by_file: HashMap<String, Vec<VectorId>>,
    pub by_kind: HashMap<String, Vec<VectorId>>,
}

impl PostingIndex {
    pub fn insert(&mut self, id: VectorId, record: &ChunkRecord) {
        self.by_file
            .entry(record.file_path.clone())
            .or_default()
            .push(id);
        self.by_kind
            .entry(record.kind.clone())
            .or_default()
            .push(id);
    }

    pub fn remove(&mut self, id: VectorId, record: &ChunkRecord) {
        remove_id(&mut self.by_file, &record.file_path, id);
        remove_id(&mut self.by_kind, &record.kind, id);
    }

    pub fn resolve<'a>(
        &self,
        file_paths: Option<&'a [String]>,
        file_path_prefix: Option<&'a str>,
        kinds: Option<&'a [String]>,
        candidate_ids: Option<&'a [VectorId]>,
        tombstones: &HashSet<VectorId>,
    ) -> Option<Vec<VectorId>> {
        let mut filters: Vec<HashSet<VectorId>> = Vec::new();

        if let Some(paths) = file_paths {
            let mut set = HashSet::new();
            for path in paths {
                if let Some(ids) = self.by_file.get(path) {
                    set.extend(ids.iter().copied());
                }
            }
            filters.push(set);
        }

        if let Some(prefix) = file_path_prefix {
            let mut set = HashSet::new();
            for (path, ids) in &self.by_file {
                if path.starts_with(prefix) {
                    set.extend(ids.iter().copied());
                }
            }
            filters.push(set);
        }

        if let Some(kind_list) = kinds {
            let mut set = HashSet::new();
            for kind in kind_list {
                if let Some(ids) = self.by_kind.get(kind) {
                    set.extend(ids.iter().copied());
                }
            }
            filters.push(set);
        }

        if let Some(ids) = candidate_ids {
            filters.push(ids.iter().copied().collect());
        }

        if filters.is_empty() {
            return None;
        }

        let mut candidates = filters[0].clone();
        for filter in filters.iter().skip(1) {
            candidates.retain(|id| filter.contains(id));
        }

        let live: Vec<VectorId> = candidates
            .into_iter()
            .filter(|id| !tombstones.contains(id))
            .collect();

        Some(live)
    }
}

fn remove_id(map: &mut HashMap<String, Vec<VectorId>>, key: &str, id: VectorId) {
    if let Some(ids) = map.get_mut(key) {
        ids.retain(|existing| *existing != id);
        if ids.is_empty() {
            map.remove(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(file: &str, kind: &str) -> ChunkRecord {
        ChunkRecord {
            text: "fn main() {}".into(),
            kind: kind.into(),
            name: None,
            file_path: file.into(),
            start_line: 1,
            end_line: 1,
        }
    }

    #[test]
    fn resolves_file_and_kind_filters_with_and_semantics() {
        let mut postings = PostingIndex::default();
        postings.insert(0, &record("src/a.rs", "Function"));
        postings.insert(1, &record("src/b.rs", "Struct"));

        let paths = vec!["src/a.rs".into()];
        let kinds = vec!["Function".into()];
        let empty = HashSet::new();

        let ids = postings
            .resolve(Some(&paths), None, Some(&kinds), None, &empty)
            .expect("filtered");
        assert_eq!(ids, vec![0]);

        let kinds = vec!["Struct".into()];
        let ids = postings
            .resolve(Some(&paths), None, Some(&kinds), None, &empty)
            .expect("filtered");
        assert!(ids.is_empty());
    }
}
