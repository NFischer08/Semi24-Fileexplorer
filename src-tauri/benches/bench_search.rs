use criterion::{criterion_group, criterion_main, Criterion};
use file_explorer_lib::db_search::run_search_logic;
use file_explorer_lib::manager::{initialize_globals, manager_make_connection_pool, VOCAB, WEIGHTS};
use std::hint::black_box;
use std::path::PathBuf;
use file_explorer_lib::config_handler::EMBEDDING_DIMENSIONS;

fn bench_search(c: &mut Criterion) {
    // --- BENCHMARK OVERRIDE ---
    // We manually populate the globals to force D150 Q8 behavior
    let weights_path = PathBuf::from("/home/magnus/RustroverProjects/Semi24-Fileexplorer/src-tauri/data/model/en_weights_D300_Q8");
    let vocab_path = PathBuf::from("/home/magnus/RustroverProjects/Semi24-Fileexplorer/src-tauri/data/model/en_vocab.json");
    let target_dim = 300;

    // 1. Force the Dimension Global
    let _ = EMBEDDING_DIMENSIONS.get_or_init(|| target_dim);

    // 2. Load the specific D150 Weights
    WEIGHTS.get_or_init(|| {
        let weights_bytes = std::fs::read(&weights_path).expect("Failed to read benchmark weights");
        let scale = 0.007874016; // Your Q8 scale
        let multiplier = scale / 127.0;

        let weights_f32: Vec<f32> = weights_bytes
            .iter()
            .map(|&b| (b as i8 as f32) * multiplier)
            .collect();

        let vocab_size = weights_f32.len() / target_dim;
        ndarray::Array2::from_shape_vec((vocab_size, target_dim), weights_f32)
            .expect("Failed to reshape benchmark weights")
    });

    // 3. Load the specific Vocab
    VOCAB.get_or_init(|| {
        let vocab_file = std::fs::File::open(&vocab_path).expect("Failed to open benchmark vocab");
        serde_json::from_reader(vocab_file).expect("Failed to parse benchmark vocab")
    });

    // Now initialize_globals() will see these are already set and do nothing
    initialize_globals();
    // --------------------------

    let pool = manager_make_connection_pool();

    if let Ok(conn) = pool.get() {
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0)).unwrap_or(0);
        eprintln!("\x1b[93m[BENCHMARK] Database entries found: {}\x1b[0m", count);
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
