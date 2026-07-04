use rek0n_db::testing::{chunk_record, unit_vector};
use rek0n_db::{CompactionPolicy, Rek0nDb, EMBEDDING_DIM, MAX_RECORD_TEXT_BYTES};

#[test]
fn rejects_invalid_compaction_policy() {
    let err = CompactionPolicy {
        dead_ratio_threshold: 1.5,
    }
    .validate()
    .expect_err("invalid threshold");
    assert!(matches!(
        err,
        rek0n_db::DbError::InvalidCompactionPolicy { .. }
    ));
}

#[test]
fn rejects_oversized_record_text() -> Result<(), rek0n_db::DbError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path())?;
    let mut record = chunk_record("src/a.rs", 1);
    record.text = "x".repeat(MAX_RECORD_TEXT_BYTES + 1);

    let err = db
        .insert_persistent(&unit_vector(0), &record)
        .expect_err("oversized text");
    assert!(matches!(err, rek0n_db::DbError::RecordTextTooLarge { .. }));
    Ok(())
}

#[test]
fn read_only_rejects_clear_staging() -> Result<(), rek0n_db::DbError> {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut db = Rek0nDb::open(dir.path())?;
        db.insert_staging(&unit_vector(0), &chunk_record("src/a.rs", 1))?;
    }

    let mut db = Rek0nDb::open_read_only(dir.path())?;
    let err = db.clear_staging().expect_err("read-only");
    assert!(matches!(err, rek0n_db::DbError::ReadOnly));
    Ok(())
}

#[test]
fn get_returns_live_vector() -> Result<(), rek0n_db::DbError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path())?;
    let vector = unit_vector(3);
    let id = db.insert_persistent(&vector, &chunk_record("src/a.rs", 1))?;

    let point = db.get(id)?;
    assert_eq!(point.id, id);
    assert_eq!(point.vector.len(), EMBEDDING_DIM);
    assert_eq!(point.record.file_path, "src/a.rs");
    Ok(())
}

#[test]
fn tombstone_single_id() -> Result<(), rek0n_db::DbError> {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path())?;
    let id = db.insert_persistent(&unit_vector(0), &chunk_record("src/a.rs", 1))?;

    assert!(db.tombstone(id)?);
    assert_eq!(db.live_persistent_count(), 0);
    Ok(())
}

#[test]
fn rejects_staging_capacity_exceeded() -> Result<(), rek0n_db::DbError> {
    use rek0n_db::MAX_STAGING_VECTORS;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path())?;
    for index in 0..MAX_STAGING_VECTORS {
        db.insert_staging(
            &unit_vector(index),
            &chunk_record("src/a.rs", index as u64),
        )?;
    }

    let err = db
        .insert_staging(
            &unit_vector(MAX_STAGING_VECTORS),
            &chunk_record("src/a.rs", MAX_STAGING_VECTORS as u64),
        )
        .expect_err("staging cap");
    assert!(matches!(
        err,
        rek0n_db::DbError::StagingCapacityExceeded { .. }
    ));
    Ok(())
}

#[test]
fn corrupt_vector_bytes_fail_open() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Rek0nDb::open(dir.path()).expect("open");
    db.insert_persistent(&unit_vector(0), &chunk_record("src/a.rs", 1))
        .expect("insert");
    drop(db);

    std::fs::write(dir.path().join("vectors.bin"), b"bad").expect("corrupt");

    let err = Rek0nDb::open(dir.path());
    assert!(matches!(err, Err(rek0n_db::DbError::CorruptVectorOffset { .. })));
}
