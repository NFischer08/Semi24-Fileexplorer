use std::sync::Mutex;
use std::sync::Arc;
use jwalk::WalkDir;
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use rayon::prelude::*;
use rusqlite::{params, MappedRows, Result, Row};
use std::{collections::HashSet, fs::{self, DirEntry}, path::{Path, PathBuf}, time::Instant,};
use std::cmp::Ordering;
use std::fmt::format;
use std::fs::File;
use std::ptr::copy;
use std::sync::mpsc;
use rayon::iter::split;
use strsim::normalized_levenshtein;
use fastembed::{TextEmbedding, InitOptions, EmbeddingModel};
use rusqlite::functions::SqlFnOutput;
use bytemuck::{cast, cast_slice};
use rayon::ThreadPool;
use tauri::Pixel;
use crossbeam::channel;

use crate::db_util::cosine_similarity;

pub fn search_database(
    connection_pool: Pool<SqliteConnectionManager>,
    search_term: &str,
    similarity_threshold: f64,
    thread_pool: &rayon::ThreadPool,
    search_path: PathBuf,
    search_file_type: &str
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

    pooled_connection.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON files(file_path)", []).expect("Indexing: ");
    pooled_connection.execute("CREATE INDEX IF NOT EXISTS idx_file_type ON files(file_type)", []).expect("Indexing: ");

    let search_file_type = search_file_type.replace(" " ,"");

    println!("query Results {}", start_time.elapsed().as_millis());

    let (tx, rx) = mpsc::channel();
    let query_thread = std::thread::spawn(move || {

        let pooled_connection = connection_pool.get().expect("get connection pool");

        let mut stmt = pooled_connection.prepare("SELECT file_path, name_embeddings FROM files WHERE  file_path LIKE ?1 AND (CASE WHEN ?2 = '' THEN 1 ELSE (',' || ?2 || ',') LIKE ('%,' || file_type || ',%') END)").map_err(|e| {
            eprintln!("Failed to prepare statement: {:?}", e);
            e
        }).expect("prepare query: ");

        let query_result = stmt.query_map(
            params![
        format!("{}%", search_path_str),
        search_file_type
    ],
            |row| Ok((
                row.get::<_, String>(0).expect("Getting row"),
                row.get::<_, Vec<u8>>(1).expect("Getting row")
            ))
        ).expect("query result: ");

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

    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::MultilingualE5Small).with_show_download_progress(true),
    ).expect("Could not create TextEmbedding");

    let mut vec_search_term = Vec::new();

    vec_search_term.push("query: ".to_string() + search_term);
    let search_vec_embedding = model.embed(vec_search_term, None).expect("Could not create TextEmbedding")[0].clone();

    let mut results: Vec<(String, f64)> = thread_pool.install(|| {
        rx.into_iter()
            .flat_map(|vec_of_paths| vec_of_paths)
            .par_bridge()  // Convert to parallel iterator
            .filter_map(|row| {
                let embedding = row.1;
                let embedding_f32 :Vec<f32> = cast_slice(&embedding).to_vec();
                let similarity = cosine_similarity(&embedding_f32, &search_vec_embedding);
                println!("similarity = {}, file_name: {}", similarity, row.0);
                if similarity > 0.85 {
                    Some((row.0, similarity.cast()))
                } else {
                    None
                }
            })
            .collect()
    });

    query_thread.join().expect("Query thread panicked").expect("Query thread panicked");

    println!("results took: {}", start_time.elapsed().as_millis());

    results.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let return_entries: Vec<DirEntry> = results.into_iter()
        .filter_map(|(s, _)| {
            let path = Path::new(&s);
            fs::read_dir(path.parent().unwrap_or(Path::new(".")))
                .ok()
                .and_then(|mut dir| dir.find(|e| e.as_ref().map(|d| d.path() == path).unwrap_or(false)))
                .and_then(|e| e.ok())
        })
        .collect();

    println!("return entries took: {}", start_time.elapsed().as_millis());

    let duration = start_time.elapsed();
    println!("Parallel search completed in {:.2?}", duration);

    Ok(return_entries)
}
