use rek0n_db::testing::{chunk_record, unit_vector};
use rek0n_db::{CompactionPolicy, Rek0nDb, EMBEDDING_DIM};

#[test]
fn indexes_searches_deletes_replaces_and_reopens() -> Result<(), rek0n_db::DbError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path())?;

    assert!(db.is_empty());

    let auth_a = unit_vector(0);
    let auth_b = unit_vector(1);
    let models = unit_vector(2);

    db.insert_persistent(&auth_a, &chunk_record("src/auth.rs", 10))?;
    db.insert_persistent(&auth_b, &chunk_record("src/auth.rs", 20))?;
    db.insert_persistent(&models, &chunk_record("src/models.rs", 1))?;

    assert_eq!(db.live_persistent_count(), 3);

    let hits = db.search(&auth_a, 2)?;
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].record.file_path, "src/auth.rs");
    assert!((hits[0].score - 1.0).abs() < f32::EPSILON);

    let removed = db.delete_by_file_path("src/auth.rs")?;
    assert_eq!(removed, 2);
    assert_eq!(db.len(), 1);

    let hits = db.search(&models, 1)?;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.file_path, "src/models.rs");

    db.replace_file(
        "src/auth.rs",
        &[(&unit_vector(40), &chunk_record("src/auth.rs", 40))],
    )?;
    assert_eq!(db.live_persistent_count(), 2);

    let hits = db.search(&unit_vector(40), 1)?;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.start_line, 40);

    let reopened = Rek0nDb::open(dir.path())?;
    assert_eq!(reopened.live_persistent_count(), 2);
    let hits = reopened.search(&unit_vector(40), 1)?;
    assert_eq!(hits[0].record.start_line, 40);

    Ok(())
}

#[test]
fn staging_flush_survives_reopen() -> Result<(), rek0n_db::DbError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path())?;

    let v0 = unit_vector(0);
    let v1 = unit_vector(1);
    db.insert_staging(&v0, &chunk_record("src/a.rs", 1))?;
    db.insert_staging(&v1, &chunk_record("src/b.rs", 2))?;

    let hits = db.search(&v0, 1)?;
    assert_eq!(hits.len(), 1);
    assert!((hits[0].score - 1.0).abs() < f32::EPSILON);

    db.flush_to_disk()?;
    assert_eq!(db.staging_count(), 0);
    assert_eq!(db.live_persistent_count(), 2);

    let reopened = Rek0nDb::open(dir.path())?;
    let hits = reopened.search(&v1, 1)?;
    assert_eq!(hits[0].record.file_path, "src/b.rs");

    Ok(())
}

#[test]
fn compact_reclaims_tombstoned_space() -> Result<(), rek0n_db::DbError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path())?.with_compaction_policy(CompactionPolicy {
        dead_ratio_threshold: 0.0,
    });

    db.insert_persistent(&unit_vector(0), &chunk_record("src/a.rs", 1))?;
    db.insert_persistent(&unit_vector(1), &chunk_record("src/b.rs", 1))?;
    db.delete_by_file_path("src/a.rs")?;

    let stats = db.compact()?;
    assert_eq!(stats.vectors_after, 1);
    assert_eq!(db.tombstone_count(), 0);
    assert_eq!(db.live_persistent_count(), 1);

    Ok(())
}

#[test]
fn reset_clears_persistent_and_staging_state() -> Result<(), rek0n_db::DbError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path())?;

    db.insert_staging(&unit_vector(0), &chunk_record("src/a.rs", 1))?;
    db.insert_persistent(&unit_vector(1), &chunk_record("src/b.rs", 2))?;
    db.reset()?;

    assert!(db.is_empty());
    assert_eq!(db.staging_count(), 0);
    assert_eq!(db.tombstone_count(), 0);

    let reopened = Rek0nDb::open(dir.path())?;
    assert!(reopened.is_empty());

    Ok(())
}

#[test]
fn rejects_mismatched_vector_dimension_on_insert() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path()).expect("open");
    let err = db
        .insert_persistent(&vec![0.0; EMBEDDING_DIM - 1], &chunk_record("src/a.rs", 1))
        .expect_err("bad dim");

    assert!(matches!(
        err,
        rek0n_db::DbError::InvalidDimension { expected, got }
        if expected == EMBEDDING_DIM && got == EMBEDDING_DIM - 1
    ));
}

#[test]
fn delete_by_file_path_removes_staging_rows() -> Result<(), rek0n_db::DbError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path())?;

    db.insert_staging(&unit_vector(0), &chunk_record("src/a.rs", 1))?;
    db.insert_staging(&unit_vector(1), &chunk_record("src/b.rs", 2))?;

    let removed = db.delete_by_file_path("src/a.rs")?;
    assert_eq!(removed, 1);
    assert_eq!(db.staging_count(), 1);

    let hits = db.search(&unit_vector(1), 1)?;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].record.file_path, "src/b.rs");

    Ok(())
}
