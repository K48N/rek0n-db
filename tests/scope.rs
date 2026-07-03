use rek0n_db::testing::chunk_record;
use rek0n_db::SearchScope;

#[test]
fn chunk_record_metadata_id_is_stable() {
    let record = chunk_record("src/auth.rs", 10);
    assert_eq!(record.metadata_id(), "src/auth.rs:10:10");
}

#[test]
fn unrestricted_scope_has_no_filters() {
    let scope = SearchScope::all();
    assert!(scope.is_unrestricted());
    assert!(scope.include_staging);
}

#[test]
fn file_path_scope_is_restricted() {
    let paths = vec!["src/a.rs".into()];
    let scope = SearchScope {
        file_paths: Some(&paths),
        ..Default::default()
    };
    assert!(!scope.is_unrestricted());
}

#[test]
fn prefix_scope_is_restricted() {
    let scope = SearchScope {
        file_path_prefix: Some("src/auth/"),
        ..Default::default()
    };
    assert!(!scope.is_unrestricted());
}

#[test]
fn candidate_id_scope_is_restricted() {
    let ids = [0_u32, 1];
    let scope = SearchScope {
        candidate_ids: Some(&ids),
        ..Default::default()
    };
    assert!(!scope.is_unrestricted());
}

#[test]
fn default_compaction_policy_matches_constant() {
    let policy = rek0n_db::CompactionPolicy::default();
    assert_eq!(
        policy.dead_ratio_threshold,
        rek0n_db::DEFAULT_COMPACT_THRESHOLD
    );
}

#[test]
fn default_ann_strategy_is_exact() {
    assert_eq!(
        rek0n_db::AnnStrategy::default(),
        rek0n_db::AnnStrategy::Exact
    );
}
