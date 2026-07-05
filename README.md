# rek0n-db

Stores and searches 384-dimensional code embeddings on disk with mmap and exact dot product.

## Overview

This crate is the lightweight local vector store rek0n uses when LanceDB churn is too heavy, especially for ephemeral MCTS branches. It memory-maps an append-only `vectors.bin`, keeps metadata in `manifest.json`, and runs exact nearest-neighbor search with optional scoped filters and IVF-lite bucketing.

Production RAG at scale should still use LanceDB through rek0n-embed. rek0n-db exists because rapid branch create/delete was thrashing my SSD during MCTS development on a ThinkPad.

## How it works

1. `Rek0nDb::open(dir)` takes an exclusive advisory lock, maps `vectors.bin`, and validates manifest integrity (record/offset parity, tombstone consistency, offset bounds).
2. Stable chunks go to the persistent tier via `insert_persistent` or `replace_file`. Deletes tombstone ids and update inverted postings by `file_path` and `kind`.
3. Ephemeral MCTS branches go to the staging tier via `insert_staging` in RAM until `flush_to_disk()` promotes them.
4. Search spans both tiers. `search()` is a full exact scan. `search_scoped()` applies filters or IVF bucket probes first, then dot product on candidates.
5. When dead bytes cross the compaction threshold, `maybe_compact()` rewrites live vectors, persists the manifest before rebuilding IVF, and keeps prior bucket settings.

## Design

**Exact search at rek0n scale.** A large repo is on the order of hundreds of thousands of chunks. Scalar dot product over a flat `f32` slice is fast enough on modern CPUs. SIMD is a future optimization if profiling says so.

**Append-only plus tombstones.** Rewriting the whole file on every delete would mirror the LanceDB churn this crate was built to avoid.

**Two tiers, two lifetimes.** Persistent mmap holds stable repository code. Staging holds branches that may disappear in seconds.

**Posting lists instead of SQL.** Code filters are predictable: path, kind, candidate ids. Inverted indexes answer those without a query engine.

## Usage

```rust
use rek0n_db::{AnnStrategy, ChunkRecord, Rek0nDb, SearchScope};

let mut db = Rek0nDb::open("~/.rek0n/vectors/my-repo")?;

let record = ChunkRecord {
    text: "fn verify(token: &str) {}".into(),
    kind: "Function".into(),
    name: Some("verify".into()),
    file_path: "src/auth.rs".into(),
    start_line: 10,
    end_line: 20,
};

db.insert_staging(&vector, &record)?;
db.flush_to_disk()?;

let hits = db.search(&query, 10)?;

let paths = vec!["src/auth.rs".to_string()];
let scope = SearchScope {
    file_paths: Some(&paths),
    include_staging: true,
    ..Default::default()
};
let scoped = db.search_scoped(&query, 10, scope, AnnStrategy::Exact)?;
```

Example:

```sh
cargo run --example index_and_search
```

## Known gaps

- IVF-lite is a simple k-means pass, not a tuned FAISS index.
- `AnnStrategy::Hnsw` is reserved until a separate `rek0n-search` crate exists.
- `compact()` rewrites the full live vector file. Rare by design, not free.
- Staging is lost on crash unless flushed. That is intentional for ephemeral branches.
- Advisory locks serialize writers on one machine. They do not federate multi-node indexes.

## License

MIT
