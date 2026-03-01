use crate::config_handler::{
    get_allowed_file_extensions, get_create_batch_size, get_embedding_dimensions,
    get_path_to_weights,
};
use crate::db_util::{convert_to_forward_slashes, full_emb, is_allowed_file, Files};
use jwalk::WalkDir;
use log::{error, info, warn};
use ndarray::Array2;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rusqlite::{params, Result};
use std::{collections::HashSet, path::PathBuf, time::Instant};

/// This Function takes in a connection pool as well as a Path as Input
/// and then recursively checks for every file / dir from the Path and adds it to the database.
/// The Vocabulary of the skip-gram model as well as the weights are being used via the pub static Once locks.
pub fn create_database(
    connection_pool: Pool<SqliteConnectionManager>,
    path: PathBuf,
) -> Result<(), String> {
    let batch_size: usize = get_create_batch_size();

    let path2 = path.clone();
    let start_time = Instant::now();

    // Starting Channel
    let (tx, rx) = crossbeam_channel::bounded(batch_size * 2);

    // Creates a Thread that genrates Hashset of every Name as well as Path from the Database
    let existing_files_thread = std::thread::spawn({
        let connection_pool = connection_pool.clone();
        move || {
            let mut existing_files = HashSet::new();
            let connection = match connection_pool.get() {
                Ok(conn) => conn,
                Err(e) => {
                    error!("Unable to get connection from pool: {e}");
                    return existing_files;
                }
            };
            let mut stmt = match connection.prepare_cached("SELECT file_name, file_path FROM files")
            {
                Ok(stmt) => stmt,
                Err(e) => {
                    error!("Failed to prepare statement: {e}");
                    return existing_files;
                }
            };
            let rows = match stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0).unwrap_or_else(|e| {
                        error!("Problem with row_get: {e}");
                        String::new()
                    }),
                    row.get::<_, String>(1).unwrap_or_else(|e| {
                        error!("Rows failed: {e}");
                        String::new()
                    }),
                ))
            }) {
                Ok(rows) => rows,
                Err(e) => {
                    error!("Failed to query result: {e}");
                    return existing_files;
                }
            };

            for (name, path) in rows.flatten() {
                existing_files.insert((name, path));
            }
            existing_files
        }
    });

    //closing Thread and getting Values
    let existing_files: HashSet<(String, String)> = match existing_files_thread.join() {
        Ok(files) => files,
        Err(_) => {
            error!("Failed to join thread for existing files.");
            return Err("Failed to join thread.".to_string());
        }
    };

    let conn = match connection_pool.get() {
        Ok(conn) => conn,
        Err(e) => {
            error!("Could not get connection from pool: {e}");
            return Err("Could not get connection from pool".to_string());
        }
    };

    // Activating Write Ahead Logging, which enables reading and writing at the same time, it should theoretically already be enabled but to be safe
    if let Err(e) = conn.execute_batch("PRAGMA journal_mode = WAL") {
        error!("Could not enable WAL: {e}");
        return Err("Could not enable WAL".to_string());
    }

    //Getting allowed file Extensions
    let allowed_file_extensions: HashSet<String> = get_allowed_file_extensions().clone();

    //Creating Thread that journals through the filesystem and sends Batches of struct Files to further processing
    let file_walking_thread = std::thread::spawn(move || {
        let mut batch: Vec<Files> = Vec::with_capacity(batch_size);
        WalkDir::new(&path)
            .follow_links(false)
            .into_iter()
            .for_each(|entry_result| {
                if let Ok(entry) = entry_result {
                    let path = entry.path();
                    //Checking that the Path is not ignored and doesn't need to be added and that it is either a directory or an allowed file extension
                    if is_allowed_file(&path, &allowed_file_extensions) {
                        let path_slashes = convert_to_forward_slashes(&path);

                        // Try to get the file stem safely
                        let file_stem = entry
                            .path()
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_string());

                        if let Some(file_name) = file_stem {
                            let file = Files {
                                id: 0,
                                file_name,
                                file_path: path_slashes,
                                file_type: if path.is_dir() {
                                    String::from("dir")
                                } else {
                                    path.extension()
                                        .and_then(|s| s.to_str())
                                        .map(String::from)
                                        .unwrap_or_else(|| String::from("binary"))
                                },
                            };
                            // Sends Batch as soon as it's Batch_Size or higher
                            batch.push(file);
                            if batch.len() >= batch_size {
                                if let Err(e) = tx.send(std::mem::replace(
                                    &mut batch,
                                    Vec::with_capacity(batch_size),
                                )) {
                                    error!("Failed to send batch: {e}");
                                    // Stop walking if receiver is gone
                                }
                            }
                        } else {
                            warn!(
                                "Warning: Couldn't get file stem for {:?}, skipping entry.",
                                entry.path()
                            );
                        }
                    }
                }
            });

        // Sends the last Batch
        if !batch.is_empty() {
            if let Err(e) = tx.send(batch) {
                error!("Failed to send final batch: {e}");
            }
        }
    });

    // Generates a Vec from every Batch of type Files, String where the Sting is the file_name without Extension
    while let Ok(batch) = rx.recv() {
        let batch_data: Vec<_> = batch
            .par_iter()
            .filter_map(|file| {
                if !existing_files.contains(&(file.file_name.clone(), file.file_path.clone())) {
                    let file_name_without_ext = file
                        .file_name
                        .split_once('.')
                        .map(|(before, _)| before.to_string())
                        .unwrap_or(file.file_name.clone());
                    Some((file.clone(), file_name_without_ext))
                } else {
                    None
                }
            })
            .collect();

        //If there is Batch Data, start Processing and get Connections and prepare SQL Statement
        if !batch_data.is_empty() {
            let mut connection = match connection_pool.get() {
                Ok(conn) => conn,
                Err(e) => {
                    error!("Unable to get connection from pool: {e}");
                    continue;
                }
            };
            let transaction = match connection.transaction() {
                Ok(tx) => tx,
                Err(e) => {
                    error!("Unable to create transaction: {e}");
                    continue;
                }
            };

            {
                //Preparing SQL Statement for Inserting Data into the DB
                let mut insert_stmt = match transaction.prepare("INSERT INTO files (file_name, file_path, file_type, name_embeddings) VALUES (?, ?, ?, ?)") {
                    Ok(stmt) => stmt,
                    Err(e) => {
                        error!("Failed to prepare insertion file: {e}");
                        continue;
                    }
                };

                //The Embedding takes up like 80% of the time per Batch

                // 1. Setup Quantization Metadata (Unchanged, but ensure get_q8_scale() is used)
                let weights_path = get_path_to_weights();
                let filename_str = weights_path.to_string_lossy();
                let dim = get_embedding_dimensions();
                let q8_scale = crate::config_handler::get_q8_scale();
                let vocab = crate::manager::VOCAB.get().unwrap(); // Get vocab for the UNK check

                // 2. Embeds the Batch conditionally
                // We use Option<Vec<f32>> to represent "meaningful" vs "junk" embeddings
                let batch_embeddings_opt: Vec<Option<Vec<f32>>> = batch_data
                    .par_iter()
                    .map(|file_data| {
                        let file_name = &file_data.1;

                        // --- NEW CHECK: Is this name just UNK tokens? ---
                        let tokens = crate::db_util::tokenize_file_name(file_name);
                        let indices = crate::db_util::tokens_to_indices(tokens, vocab);

                        if indices.is_empty() || indices.iter().all(|&idx| idx == 0) {
                            None // Mark as junk/NULL
                        } else {
                            Some(full_emb(file_name)) // Generate real embedding
                        }
                    })
                    .collect();

                // 3. Transform into Vec<Option<Vec<u8>>>
                // This allows the DB layer to insert NULL for the None variants
                let embeddings_u8: Vec<Option<Vec<u8>>> = batch_embeddings_opt
                    .into_iter()
                    .map(|opt_emb| {
                        let embedding_vec = opt_emb?; // If None, returns None for the whole map

                        let row = if filename_str.contains("_Q8") {
                            let mut r = Vec::with_capacity(dim);
                            for f in embedding_vec {
                                let quantized = (f * 127.0 / q8_scale).clamp(-128.0, 127.0) as i8;
                                r.push(quantized as u8);
                            }
                            r
                        } else if filename_str.contains("_Q16") {
                            let mut r = Vec::with_capacity(dim * 2);
                            for f in embedding_vec {
                                r.extend_from_slice(&half::f16::from_f32(f).to_le_bytes());
                            }
                            r
                        } else {
                            embedding_vec.iter().flat_map(|f| f.to_le_bytes()).collect()
                        };

                        Some(row)
                    })
                    .collect();

                // 4. Batch Data Insertion
                for (c, file_data) in batch_data.iter().enumerate() {
                    let file = &file_data.0;
                    if let Some(vec) = embeddings_u8.get(c) {
                        if let Err(e) = insert_stmt.execute(params![
                            file.file_name,
                            file.file_path,
                            file.file_type,
                            vec
                        ]) {
                            error!("Could not insert file {file:?}: {e}");
                        }
                    }
                }
            }
            if let Err(e) = transaction.commit() {
                error!("Unable to commit transaction: {e}");
            }
        }
    }
    if file_walking_thread.join().is_err() {
        error!("Failed to join file walking thread.");
        return Err("Failed to join file walking thread.".to_string());
    }

    info!(
        "Database population for {:?} took {}ms",
        path2,
        start_time.elapsed().as_millis()
    );
    Ok(())
}
