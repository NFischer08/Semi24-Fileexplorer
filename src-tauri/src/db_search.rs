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
use std::collections::HashMap;
use tch::{CModule, Kind};
use strsim::normalized_levenshtein;
use crate::manager::THREAD_POOL;

pub fn search_database(
    connection_pool: Pool<SqliteConnectionManager>,
    search_term: &str,
    search_path: PathBuf,
    search_file_type: &str,
    model: &CModule,
    num_results_embeddings: usize,
    num_results_levenhstein: usize,
    vocab: &HashMap<String, usize>,
) -> Result<Vec<DirEntry>> {
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

    let (tx, rx) = crossbeam_channel::unbounded();

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

    let search_norm: f32 = embedded_vec_f32.iter()
        .fold(0.0, |acc, &x| acc + x * x)
        .sqrt();

    let results_start = Instant::now();

    let results: Vec<(String, f32, f64)> = rx.into_iter()
            .par_bridge()
            .flat_map(|vec_of_paths| vec_of_paths)
            .filter_map(|row| {
                let path = PathBuf::from(&row.0);
                let file_name = path.file_stem()
                    .and_then(|os_str| os_str.to_str())
                    .unwrap_or("invalid_unicode");

                let vec_similarity = cosine_similarity(
                    cast_slice::<u8, f32>(&row.1),
                    search_norm,
                    &embedded_vec_f32
                );
                let normalized_levenhstein_dist = normalized_levenshtein(file_name, search_term);

                Some((row.0.clone(), vec_similarity, normalized_levenhstein_dist))
            })
        .collect();

    println!("Finished results in : {:?} ", results_start.elapsed());

    query_thread
        .join()
        .expect("Query thread panicked")
        .expect("Query thread panicked");

    let (mut results_embedding, mut results_levenhstein): (Vec<_>, Vec<_>) = results
        .into_iter()
        .map(|(s, f, u)| {
            let s2 = s.clone();  // Single clone operation
            ((s, f), (s2, u))
        })
        .unzip();

    results_levenhstein.par_sort_by(|a, b|
        b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
    );


    results_levenhstein.truncate(num_results_levenhstein);

    //println!("results levenhstein {:?}", &results_levenhstein);

    results_embedding.par_sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
    });

    results_embedding.truncate(num_results_embeddings);

    //println!("results embeddings {:?}", &results_embedding);


    let results: Vec<String> = results_levenhstein
        .into_iter()
        .map(|(s, _)| s)
        .chain(results_embedding.into_iter().map(|(s, _)| s))
        .collect();

    let return_entries: Vec<DirEntry> = results
        .into_iter()
        .filter_map(|s| {
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
        .collect();

    println!("search took: {:?}", start_time.elapsed().as_millis());

    Ok(return_entries)
}
