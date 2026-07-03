use std::path::PathBuf;

use rek0n_db::{AnnStrategy, ChunkRecord, Rek0nDb, SearchScope, EMBEDDING_DIM};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db_dir = db_path();
    println!("db: {}", db_dir.display());

    let mut db = Rek0nDb::open(&db_dir)?;
    db.reset()?;

    index_repo(&mut db)?;
    print_search(&db, "jwt auth", 0)?;
    print_scoped_search(&db, 1)?;
    demo_staging(&mut db)?;

    Ok(())
}

fn db_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/data/demo-repo")
}

fn index_repo(db: &mut Rek0nDb) -> Result<(), Box<dyn std::error::Error>> {
    let chunks = [
        chunk("src/auth.rs", "Function", "verify jwt token", 10, 0),
        chunk("src/auth.rs", "Function", "hash password bcrypt", 20, 1),
        chunk("src/models.rs", "Struct", "struct User id email", 1, 2),
    ];

    for (record, vector) in chunks {
        db.insert_persistent(&vector, &record)?;
        println!(
            "indexed: {}:{} {} ({})",
            record.file_path, record.start_line, record.kind, record.text
        );
    }

    println!("live vectors: {}", db.live_persistent_count());
    Ok(())
}

fn print_search(
    db: &Rek0nDb,
    label: &str,
    active: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let query = unit_vector(active);
    let hits = db.search(&query, 2)?;

    println!("search ({label}):");
    for hit in hits {
        println!(
            "  {:.3}  {}:{}  {}",
            hit.score, hit.record.file_path, hit.record.start_line, hit.record.text
        );
    }

    Ok(())
}

fn print_scoped_search(db: &Rek0nDb, active: usize) -> Result<(), Box<dyn std::error::Error>> {
    let query = unit_vector(active);
    let paths = vec!["src/auth.rs".to_string()];
    let scope = SearchScope {
        file_paths: Some(&paths),
        include_staging: false,
        ..Default::default()
    };

    let hits = db.search_scoped(&query, 3, scope, AnnStrategy::Exact)?;

    println!("scoped search (src/auth.rs):");
    for hit in hits {
        println!(
            "  {:.3}  {}:{}  {}",
            hit.score, hit.record.file_path, hit.record.start_line, hit.record.text
        );
    }

    Ok(())
}

fn demo_staging(db: &mut Rek0nDb) -> Result<(), Box<dyn std::error::Error>> {
    let (record, vector) = chunk("src/auth.rs", "Function", "rotate session token", 40, 40);
    db.insert_staging(&vector, &record)?;
    println!("staging: {} ephemeral vector(s)", db.staging_count());

    print_search(db, "staging branch", 40)?;

    db.flush_to_disk()?;
    println!(
        "flushed staging to disk (live: {})",
        db.live_persistent_count()
    );

    Ok(())
}

fn chunk(
    file_path: &str,
    kind: &str,
    text: &str,
    line: u64,
    vector_axis: usize,
) -> (ChunkRecord, Vec<f32>) {
    let record = ChunkRecord {
        text: text.into(),
        kind: kind.into(),
        name: Some("demo".into()),
        file_path: file_path.into(),
        start_line: line,
        end_line: line,
    };
    (record, unit_vector(vector_axis))
}

fn unit_vector(active: usize) -> Vec<f32> {
    let mut vector = vec![0.0_f32; EMBEDDING_DIM];
    vector[active % EMBEDDING_DIM] = 1.0;
    vector
}
