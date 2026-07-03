use rek0n_db::testing::{chunk_record, chunk_record_with_kind, unit_vector};
use rek0n_db::{AnnStrategy, DbError, Rek0nDb, SearchScope, EMBEDDING_DIM};

#[test]
fn scoped_search_filters_by_file_path() -> Result<(), DbError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path())?;

    db.insert_persistent(&unit_vector(0), &chunk_record("src/a.rs", 1))?;
    db.insert_persistent(&unit_vector(1), &chunk_record("src/b.rs", 1))?;

    let paths = vec!["src/a.rs".into()];
    let scope = SearchScope {
        file_paths: Some(&paths),
        include_staging: false,
        ..Default::default()
    };

    let hits = db.search_scoped(&unit_vector(0), 5, scope, AnnStrategy::Exact)?;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.file_path, "src/a.rs");

    Ok(())
}

#[test]
fn scoped_search_filters_by_kind() -> Result<(), DbError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path())?;

    db.insert_persistent(
        &unit_vector(0),
        &chunk_record_with_kind("src/a.rs", "Function", 1),
    )?;
    db.insert_persistent(
        &unit_vector(1),
        &chunk_record_with_kind("src/a.rs", "Struct", 2),
    )?;

    let kinds = vec!["Struct".into()];
    let scope = SearchScope {
        kinds: Some(&kinds),
        include_staging: false,
        ..Default::default()
    };

    let hits = db.search_scoped(&unit_vector(1), 5, scope, AnnStrategy::Exact)?;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.kind, "Struct");

    Ok(())
}

#[test]
fn ivf_search_returns_ranked_hits() -> Result<(), DbError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path())?;

    for index in 0..8 {
        db.insert_persistent(
            &unit_vector(index),
            &chunk_record(&format!("src/f{index}.rs"), index as u64),
        )?;
    }

    db.build_ivf_index(4, 1)?;

    let hits = db.search_scoped(
        &unit_vector(0),
        5,
        SearchScope::all(),
        AnnStrategy::Ivf { probe_buckets: 1 },
    )?;
    assert!(!hits.is_empty());
    assert!(hits.len() <= 5);
    assert_eq!(hits[0].record.file_path, "src/f0.rs");

    Ok(())
}

#[test]
fn rejects_wrong_query_dimension() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Rek0nDb::open(dir.path()).expect("open");
    let err = db.search(&[0.0; 3], 1).expect_err("bad dim");
    assert!(matches!(err, DbError::InvalidQuery { .. }));
}

#[test]
fn rejects_zero_search_limit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Rek0nDb::open(dir.path()).expect("open");
    let err = db.search(&unit_vector(0), 0).expect_err("zero limit");
    assert!(matches!(err, DbError::InvalidSearchLimit));
}

#[test]
fn rejects_ivf_before_index_is_built() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path()).expect("open");
    db.insert_persistent(&unit_vector(0), &chunk_record("src/a.rs", 1))
        .expect("insert");

    let err = db
        .search_scoped(
            &unit_vector(0),
            1,
            SearchScope::all(),
            AnnStrategy::Ivf { probe_buckets: 1 },
        )
        .expect_err("ivf missing");
    assert!(matches!(err, DbError::IvfNotBuilt));
}

#[test]
fn rejects_hnsw_until_rek0n_search_exists() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Rek0nDb::open(dir.path()).expect("open");
    let err = db
        .search_scoped(
            &vec![0.0; EMBEDDING_DIM],
            1,
            SearchScope::all(),
            AnnStrategy::Hnsw { ef_search: 64 },
        )
        .expect_err("hnsw reserved");
    assert!(matches!(err, DbError::HnswNotBuilt));
}

#[test]
fn staging_only_hits_respect_include_staging_flag() -> Result<(), DbError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path())?;

    db.insert_staging(&unit_vector(0), &chunk_record("src/a.rs", 1))?;

    let scope = SearchScope {
        include_staging: false,
        ..Default::default()
    };
    let hits = db.search_scoped(&unit_vector(0), 5, scope, AnnStrategy::Exact)?;
    assert!(hits.is_empty());

    let hits = db.search(&unit_vector(0), 5)?;
    assert_eq!(hits.len(), 1);

    Ok(())
}
