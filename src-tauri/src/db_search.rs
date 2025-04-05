use crate::db_util::{cosine_similarity, load_vocab, tokenize_file_name, tokens_to_indices};
use bytemuck::cast_slice;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::prelude::*;
use rayon::ThreadPool;
use rusqlite::{params, Result};
use std::cmp::Ordering;
use std::sync::mpsc;
use std::{
    fs::{self, DirEntry},
    path::{Path, PathBuf},
    time::Instant,
};
use tch::{CModule, Kind};

pub fn search_database(
    connection_pool: Pool<SqliteConnectionManager>,
    search_term: &str,
    thread_pool: &ThreadPool,
    search_path: PathBuf,
    search_file_type: &str,
    model: &CModule,
    num_results: usize,
) -> Result<Vec<DirEntry>> {
    let pooled_connection = connection_pool.get().expect("get connection pool");
    const BATCH_SIZE: usize = 1000; // Adjust this value as needed

    let start_time = Instant::now();

    // Convert searchpath to a string
    let search_path_str = if cfg!(windows) && search_path.to_str().unwrap_or("") == "/" {
        String::new()
    } else {
        search_path.to_str().unwrap_or("").to_string()
    };

    pooled_connection
        .execute(
            "CREATE INDEX IF NOT EXISTS idx_file_path ON files(file_path)",
            [],
        )
        .expect("Indexing: ");
    pooled_connection
        .execute(
            "CREATE INDEX IF NOT EXISTS idx_file_type ON files(file_type)",
            [],
        )
        .expect("Indexing: ");

    let search_file_type = search_file_type.replace(" ", "");

    let (tx, rx) = mpsc::channel();
    let query_thread = std::thread::spawn(move || {
        let pooled_connection = connection_pool.get().expect("get connection pool");

        let mut stmt = pooled_connection.prepare("SELECT file_path, name_embeddings FROM files WHERE  file_path LIKE ?1 AND (CASE WHEN ?2 = '' THEN 1 ELSE (',' || ?2 || ',') LIKE ('%,' || file_type || ',%') END)").map_err(|e| {
            eprintln!("Failed to prepare statement: {:?}", e);
            e
        }).expect("prepare query: ");

        let query_result = stmt
            .query_map(
                params![format!("{}%", search_path_str), search_file_type],
                |row| {
                    Ok((
                        row.get::<_, String>(0).expect("Getting row"),
                        row.get::<_, Vec<u8>>(1).expect("Getting row"),
                    ))
                },
            )
            .expect("query result: ");

        let mut batch = Vec::with_capacity(BATCH_SIZE);
        for result in query_result {
            match result {
                Ok(file_path) => {
                    batch.push(file_path);
                    if batch.len() >= BATCH_SIZE {
                        tx.send(batch).expect("Failed to send batch");
                        batch = Vec::with_capacity(BATCH_SIZE);
                    }
                }
                Err(e) => {
                    eprintln!("Error in query result: {:?}", e);
                    return Err(e);
                }
            }
        }

        // Send any remaining items
        if !batch.is_empty() {
            tx.send(batch).expect("Failed to send final batch");
        }
        Ok(())
    });

    let mut vec_search_term = Vec::new();

    vec_search_term.push("query: ".to_string() + search_term);

    let vocab = load_vocab("src-tauri/src/neural_network/vocab.json");

    let tokenized_searchterm = tokenize_file_name(search_term);
    let indiced_searchterm: Vec<i64> = tokens_to_indices(tokenized_searchterm, &vocab)
        .into_iter()
        .map(|x| x as i64)
        .collect();

    let search_tensor = tch::Tensor::from_slice(&indiced_searchterm)
        .to_kind(Kind::Int64)
        .unsqueeze(0);

    let embedded_tensor = model
        .method_ts("get_embedding", &[search_tensor])
        .expect("Batch embedding lookup failed");

    let embedded_f32 = embedded_tensor.to_kind(Kind::Float); // Ensure Float32 type

    let embedded_vec_f32: Vec<f32> =
        Vec::try_from(embedded_f32.flatten(0, -1)).expect("Can't convert vector to f32");

    let mut results: Vec<(String, f64)> = thread_pool.install(|| {
        rx.into_iter()
            .flat_map(|vec_of_paths| vec_of_paths)
            .par_bridge() // Convert to parallel iterator
            .filter_map(|row| {
                let embedding = row.1;
                let embedding_f32 = cast_slice(&embedding).to_vec();
                let similarity: f32 = cosine_similarity(&embedding_f32, &embedded_vec_f32); //Similarity of 1 = full match
                Some((row.0, similarity as _))
            })
            .collect()
    });

    query_thread
        .join()
        .expect("Query thread panicked")
        .expect("Query thread panicked");

    results.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    results.truncate(num_results);

    let return_entries: Vec<DirEntry> = results
        .into_iter()
        .filter_map(|(s, _)| {
            let path = Path::new(&s);
            if let Some(parent) = path.parent() {
                if let Ok(dir) = fs::read_dir(parent) {
                    // Use find_map to directly return the matching entry
                    return dir
                        .filter_map(Result::ok)
                        .find(|entry| entry.path() == path);
                }
            }

            None
        })
        .collect();

    println!("search took: {:?}", start_time.elapsed().as_millis());

    Ok(return_entries)
}
