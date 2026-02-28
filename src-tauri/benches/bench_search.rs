use criterion::{criterion_group, criterion_main, Criterion};
use file_explorer_lib::db_search::run_search_logic;
use file_explorer_lib::manager::{initialize_globals, manager_make_connection_pool};
use std::hint::black_box;
use std::path::PathBuf;

fn bench_search(c: &mut Criterion) {
    initialize_globals();
    let pool = manager_make_connection_pool();

    // Verification of DB state
    if let Ok(conn) = pool.get() {
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))
            .unwrap_or(0);
        println!("\n[BENCHMARK] Total entries in database: {}\n", count);
    }

    c.bench_function("search_database_performance", |b| {
        b.iter(|| {
            run_search_logic(
                black_box(pool.clone()),
                black_box("test"),
                black_box(PathBuf::from("/")),
                black_box("pdf".to_string()),
                black_box(10),
                black_box(10),
                |_| {},
            );
        })
    });
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
