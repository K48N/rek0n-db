use std::thread;
use std::time::Duration;

use rek0n_db::testing::{chunk_record, unit_vector};
use rek0n_db::{DbLockOptions, Rek0nDb, DEFAULT_LOCK_TIMEOUT};

#[test]
fn read_only_open_rejects_mutations() -> Result<(), rek0n_db::DbError> {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut db = Rek0nDb::open(dir.path())?;
        db.insert_persistent(&unit_vector(0), &chunk_record("src/a.rs", 1))?;
    }

    let mut db = Rek0nDb::open_read_only(dir.path())?;
    assert!(db.is_read_only());

    let err = db
        .insert_staging(&unit_vector(1), &chunk_record("src/b.rs", 2))
        .expect_err("read-only");
    assert!(matches!(err, rek0n_db::DbError::ReadOnly));

    Ok(())
}

#[test]
fn exclusive_lock_blocks_second_writer() {
    let dir = tempfile::tempdir().expect("tempdir");
    let _first = Rek0nDb::open(dir.path()).expect("first open");

    let err = Rek0nDb::open_with_options(dir.path(), DbLockOptions::try_exclusive_once());
    assert!(matches!(err, Err(rek0n_db::DbError::LockTimeout { .. })));
}

#[test]
fn shared_lock_allows_concurrent_readers() {
    let dir = tempfile::tempdir().expect("tempdir");
    Rek0nDb::open(dir.path())
        .expect("writer")
        .insert_persistent(&unit_vector(0), &chunk_record("src/a.rs", 1))
        .expect("insert");

    let _reader_a = Rek0nDb::open_read_only(dir.path()).expect("reader a");
    let _reader_b = Rek0nDb::open_read_only(dir.path()).expect("reader b");
}

#[test]
fn writer_waits_for_lock_then_succeeds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_path_buf();
    let holder = Rek0nDb::open(&path).expect("holder");

    let path_clone = path.clone();
    let join = thread::spawn(move || {
        Rek0nDb::open_with_options(path_clone, DbLockOptions::exclusive(Duration::from_secs(2)))
    });

    thread::sleep(Duration::from_millis(100));
    drop(holder);

    join.join()
        .expect("join")
        .expect("second writer should acquire lock after release");
}

#[test]
fn default_lock_timeout_is_thirty_seconds() {
    assert_eq!(DEFAULT_LOCK_TIMEOUT, Duration::from_secs(30));
}
