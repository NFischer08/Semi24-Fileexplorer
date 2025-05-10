use crate::config_handler::get_search_batch_size;
use crate::db_util::{bytes_to_vec, cosine_similarity, full_emb, tokenize_file_name, tokens_to_indices};
use crate::manager::{build_struct, AppState, VOCAB, WEIGHTS};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::prelude::*;
use rusqlite::{params, Result};
use std::cmp::Ordering;
use std::{
    fs::{self, DirEntry},
    path::{Path, PathBuf},
    time::Instant,
};
use strsim::normalized_levenshtein;
use tauri::{Emitter, State};

/// Searches for similar File names in the Database via Levenhstein and a custome skip-gram model,
/// it uses connection_pool, search_term, search_path, search_file_type, num_results_lev, num_results_emb and state
pub fn search_database(
    connection_pool: Pool<SqliteConnectionManager>,
    search_term: &str,
    search_path: PathBuf,
    search_file_type: &str,
    num_results_embeddings: usize,
    num_results_levenhstein: usize,
    state: State<AppState>,
) {
    // Getting a Pooled Connetion
    let pooled_connection = connection_pool.get().expect("get connection pool");
    let batch_size: usize = get_search_batch_size();

    let start_time = Instant::now();

    //Setting Search Path to "" for searching everythin
    let search_path_str = if cfg!(windows) && search_path.to_str().unwrap_or("") == "/" {
        String::new()
    } else {
        search_path.to_str().unwrap_or("").to_string()
    };

    //Making sure relecant Collums are Indexed
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

    //Making sure there are no Spaces in file_types
    let search_file_type = search_file_type.replace(" ", "");

    //Creating channel
    let (sender, receiver) = crossbeam_channel::unbounded();

    //Creating Thread that gets relevant Data from the Database, it already sorts for file_type and Path via the SQL Statement
    let query_thread = std::thread::spawn(move || {
        let mut pooled_connection = connection_pool.get().expect("Failed to get connection");
        let tx = pooled_connection
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

            //Getting the rows file_path and name_embeddings
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

        //Pre AlLocating the Memory for batch_emb
        let mut batch_emb: Vec<(String, Vec<u8>)> = Vec::with_capacity(batch_size);

        //Splitting the Data into Batches which are sent to processing
        for path_emb in path_embs {
            batch_emb.push(path_emb);

            if batch_emb.len() >= batch_size {
                sender
                    .send(std::mem::replace(
                        &mut batch_emb, // Changed from `batch` to `batch_emb`
                        Vec::with_capacity(batch_size),
                    ))
                    .expect("Failed to send result");
            }
        }

        //Sending last Batch
        if !batch_emb.is_empty() {
            sender.send(batch_emb).expect("Receiving thread: drop");
        }
    });

    // Creating the Vec
    let embedded_vec_f32 = full_emb(search_term);

    let mut search_query: Vec<(String, Vec<u8>)> = Vec::new();

    // Computes the Levenhstein distance / similarity as well as builds up a Vec of every batch
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

    // Transform the results into DireEntrys, sorts them and only give back the num_results_lev best results
    let ret_lev_dir = return_entries(results_lev, num_results_levenhstein);

    // Transforms the results into FileDataFormatted which the FrontEnd uses
    let ret_lev = build_struct(ret_lev_dir);

    // Sends the result to the FrontEnd via a Tauri Signal
    state.handle.emit("search-finnished", &ret_lev).unwrap();
    println!(
        "levenhstein-finnished {:?}",
        start_time.elapsed().as_millis()
    );

    // Computes the Embedding similarity / Cosine Similarity
    let results_emb: Vec<(String, f32)> = search_query
        .into_par_iter() // Iterate over Vec<(String, Vec<u8>)>
        .map(|(path, embedding)| {
            // Destructure each tuple
            let embedding_f32 = bytes_to_vec(&embedding); //THIS IS CORRECT
            let similarity = cosine_similarity(&embedding_f32, &embedded_vec_f32);
            (path, similarity) // Return new tuple
        })
        .collect(); // Collect results

    query_thread.join().expect("Query thread panicked");

    // Transform the results into DireEntrys, sorts them and only give back the num_results_emb best results
    let ret_emb_dir = return_entries(results_emb, num_results_embeddings);

    // Transforms the results into FileDataFormatted which the FrontEnd uses
    let ret_emb = build_struct(ret_emb_dir);

    // Adds the results embedding and levenhstein together
    let ret = [ret_lev, ret_emb].concat();

    // Sends final results to FrontEnd
    state.handle.emit("search-finnished", &ret).unwrap();
    println!("embedding-finnished {:?}", start_time.elapsed().as_millis());
    println!("search took: {:?}", start_time.elapsed().as_millis());
}

/// Support Funciton for searching which only gives back the best results in form of DirEntries
fn return_entries(mut similarity_values: Vec<(String, f32)>, num_ret: usize) -> Vec<DirEntry> {
    similarity_values.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    similarity_values.truncate(num_ret);

    println!("Similarity values {:?}", similarity_values);
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
