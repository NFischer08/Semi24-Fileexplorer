use criterion::{criterion_group, criterion_main, Criterion};
use file_explorer_lib::db_create::create_database;
use file_explorer_lib::manager::manager_make_connection_pool;
use std::fs::File;
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use tempfile::tempdir;

fn bench_create_db(c: &mut Criterion) {
    file_explorer_lib::manager::initialize_globals();
    // 1. Setup a dummy directory structure
    let dir = tempdir().expect("Failed to create temp dir");
    for i in 0..1000 {
        let file_path = dir.path().join(format!("test_file_{}.pdf", i));
        let mut f = File::create(file_path).unwrap();
        writeln!(f, "dummy content").unwrap();
    }

    c.bench_function("create_database_1000_files", |b| {
        b.iter(|| {
            // We recreate the pool/db each time to ensure we aren't
            // just hitting the "existing_files" HashSet bypass.
            let pool = manager_make_connection_pool();

            let result = create_database(black_box(pool), black_box(dir.path().to_path_buf()));

            assert!(result.is_ok());
        })
    });
}

criterion_group!(benches, bench_create_db);
criterion_main!(benches);
