use crate::db_util::{cosine_similarity, tokenize_file_name, tokens_to_indices};
use crate::manager::{build_struct, AppState};
use bytemuck::cast_slice;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::prelude::*;
use rusqlite::{params, MappedRows, Result, Row};
use std::any::Any;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::{
    fs::{self, DirEntry},
    path::{Path, PathBuf},
    time::Instant,
};
use strsim::normalized_levenshtein;
use tauri::{Emitter, State};
use tch::{CModule, Kind};

pub fn search_database(
    connection_pool: Pool<SqliteConnectionManager>,
    search_term: &str,
    search_path: PathBuf,
    search_file_type: &str,
    model: &CModule,
    vocab: &HashMap<String, usize>,
    num_results_embeddings: usize,
    num_results_levenhstein: usize,
    state: State<AppState>,
) -> () {
    let pooled_connection = connection_pool.get().expect("get connection pool");
    const BATCH_SIZE: usize = 1000; // Adjust this value as needed

    let start_time = Instant::now();

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

    let (sender, receiver) = crossbeam_channel::unbounded();

    let query_thread = std::thread::spawn(move || {
        let mut pooled_connection = connection_pool.get().expect("Failed to get connection");
        let mut tx = pooled_connection
            .transaction()
            .expect("Failed to begin transaction");

        let path_embs: Vec<(String, Vec<u8>)> = {
            let mut stmt = tx
                .prepare_cached(
                    r#"
        SELECT file_path, name_embeddings
        FROM files
        WHERE file_path LIKE ?1
        AND (?2 = '' OR file_type = ?2)
        "#,
                )
                .expect("Failed to prepare statement");

            let search_pattern = format!("{}%", search_path_str);
            stmt.query_map(params![search_pattern, search_file_type], |row| {
                Ok((
                    row.get::<_, String>("file_path")?,
                    row.get::<_, Vec<u8>>("name_embeddings")?,
                ))
            })
            .expect("Query execution failed")
            .collect::<Result<Vec<_>, _>>()
            .expect("Result collection failed")
        }; // TODO improve Perf of code Section above, takes up more than half of search

        tx.commit().expect("Failed to commit transaction");

        let mut batch_emb: Vec<(String, Vec<u8>)> = Vec::with_capacity(BATCH_SIZE);
        for path_emb in path_embs {
            batch_emb.push(path_emb);

            if batch_emb.len() >= BATCH_SIZE {
                sender
                    .send(std::mem::replace(
                        &mut batch_emb, // Changed from `batch` to `batch_emb`
                        Vec::with_capacity(BATCH_SIZE),
                    ))
                    .expect("Failed to send result");
            }
        }

        if !batch_emb.is_empty() {
            sender.send(batch_emb).expect("Receiving thread: drop");
        }

        //println!("Processed rows in {}ms", start_time.elapsed().as_millis());
    });

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

    let search_norm: f32 = embedded_vec_f32
        .iter()
        .fold(0.0, |acc, &x| acc + x * x)
        .sqrt();

    // Store both String and Vec<u8> pairs
    // Wrap in Arc<Mutex> for thread-safe mutation
    let mut search_query: Vec<(String, Vec<u8>)> = Vec::new();

    let results_lev: Vec<(String, f32)> = receiver
        .into_iter() // Convert to parallel iterator
        .flat_map(|batch| {
            // Lock and extend atomically
            search_query.extend(batch.clone());
            batch.into_iter() // Process in parallel
        })
        .par_bridge()
        .map(|(row, _)| {
            let path = PathBuf::from(&row);
            let file_name = path
                .file_stem()
                .and_then(|os_str| os_str.to_str())
                .unwrap_or("invalid_unicode");

            (row, normalized_levenshtein(file_name, search_term) as f32)
        })
        .collect();

    let ret_lev_dir = return_entries(results_lev, num_results_levenhstein);
    let ret_lev = build_struct(ret_lev_dir);
    state.handle.emit("search-finnished", &ret_lev).unwrap();
    println!(
        "levenhstein-finnished {:?}",
        start_time.elapsed().as_millis()
    );

    let results_emb: Vec<(String, f32)> = search_query
        .into_par_iter() // Iterate over Vec<(String, Vec<u8>)>
        .map(|(path, embedding)| {
            // Destructure each tuple
            let embedding_f32 = cast_slice::<u8, f32>(&embedding);
            let similarity = cosine_similarity(embedding_f32, search_norm, &embedded_vec_f32);
            (path, similarity) // Return new tuple
        })
        .collect(); // Collect results

    query_thread.join().expect("Query thread panicked");

    let ret_emb_dir = return_entries(results_emb, num_results_embeddings);
    let ret_emb = build_struct(ret_emb_dir);
    let ret = [ret_lev, ret_emb].concat();
    state.handle.emit("search-finnished", &ret).unwrap();
    println!("embedding-finnished {:?}", start_time.elapsed().as_millis());

    println!("search took: {:?}", start_time.elapsed().as_millis());
}

fn return_entries(mut similarity_values: Vec<(String, f32)>, num_ret: usize) -> Vec<DirEntry> {
    similarity_values.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    similarity_values.truncate(num_ret);

    similarity_values
        .into_par_iter()
        .filter_map(|(s, _)| {
            let path = Path::new(&s);
            if let Some(parent) = path.parent() {
                if let Ok(dir) = fs::read_dir(parent) {
                    return dir
                        .filter_map(Result::ok)
                        .find(|entry| entry.path() == path);
                }
            }
            None
        })
        .collect()
}
