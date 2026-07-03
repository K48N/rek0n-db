# rek0n-db

Part of my project rek0n. Stores and searches 384-d code embeddings locally: mmap on disk, exact dot-product in RAM, no LanceDB.

## What it is

A Rust library that is the "SQLite of vector databases." It takes L2-normalized `f32` vectors from rek0n-embed, persists them as a flat append-only `.bin` file, and runs exact nearest-neighbor search with optional scoped filters and IVF-lite bucketing. A separate in-memory staging tier holds ephemeral MCTS branches without touching the SSD.

## Please Note

In a production deployment, using an established tool like LanceDB is the correct choice, and rek0n-embed is already wired to support it. However, I built this entire project on a hardware-constrained ThinkPad, and compile times mattered a lot to me during development.

During testing of my Monte Carlo Tree Search algorithm, standard LanceDB row deletes were thrashing my local SSD due to the rapid creation and pruning of ephemeral branches. I built rek0n-db as a custom, lightweight memory-mapped development tier specifically to solve that local bottleneck NOT as a LanceDB replacement.

## How it works

1. `Rek0nDb::open(dir)` memory-maps `vectors.bin` and loads `manifest.json` (records, byte offsets, tombstones, posting lists, optional IVF assignments).
2. Stable chunks go to the **persistent tier** via `insert_persistent` or `replace_file`: vectors append at EOF, and deletes tombstone ids and update inverted postings by `file_path` / `kind`.
3. In-flight MCTS branches go to the **staging tier** via `insert_staging`: pure RAM until `flush_to_disk()` promotes them.
4. Search spans both tiers. `search()` runs Tier 0 exact scan. `search_scoped()` applies `SearchScope` filters (Tier 1) or probes IVF centroid buckets first (Tier 2), then exact dot-product on candidates.
5. When tombstoned bytes exceed the compaction threshold, `maybe_compact()` rewrites live vectors into a fresh `vectors.bin` and clears dead weight.

## Why it's built this way

**Exact search is enough at rek0n scale.** A large repo is ~500k tree-sitter chunks. SIMD dot-product over a flat `f32` array is milliseconds. HNSW and Lance IVF-PQ solve billions of vectors; rek0n does not and will probably not have billions. I will still eventually build a minimal, separate crate called rek0n-db-search to implement HNSW.

**Append-only + tombstones, not rewrite-on-delete.** Lance-style row deletes on every file save thrash disk. Tombstoning plus lazy compaction keeps per-file `replace_file` at O(chunks in file) until compaction is actually needed.

**Posting lists instead of SQL.** Code RAG filters are predictable: `file_path`, `kind`, MCTS candidate ids. Inverted indexes answer those without a query engine or Arrow schemas.

**Two tiers for two lifetimes.** Persistent mmap holds stable repository code. Staging holds branches that may be pruned in seconds. Mixing them in one disk-backed store is the wrong abstraction for Phase 3 MCTS.

**Minimal dependency stack.** Only `memmap2`, `serde`, `serde_json`, and `thiserror`. No Apache Arrow, no background vector service, compiles in seconds.

## Shortcomings

- No built-in file locking: rek0n-embed owns exclusive/shared locks around writes.
- IVF-lite is k-means-lite with flat centroids, not a tuned FAISS index; build it when profiling says brute force is slow.
- `AnnStrategy::Hnsw` is reserved: it returns `HnswNotBuilt` until the `rek0n-db-search` crate exists.
- `compact()` rewrites the full live vector file, which is rare by design but not free.
- Staging is lost on process crash unless flushed; that is intentional for ephemeral branches.
- Phase 5 swarm federation is out of scope; each node carries its own db directory today.

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
let hits = db.search_scoped(&query, 10, scope, AnnStrategy::Exact)?;
```

See `examples/index_and_search.rs` for a full index → search → staging → flush flow:

```sh
cargo run --example index_and_search
```

## License

MIT
