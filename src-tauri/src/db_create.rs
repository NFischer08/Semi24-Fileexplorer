use crate::config_handler::{get_allowed_file_extensions, get_create_batch_size};
use crate::db_util::{
    convert_to_forward_slashes, is_allowed_file, should_ignore_path, tokenize_file_name,
    tokens_to_indices, Files,
};
use jwalk::WalkDir;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::prelude::*;
use rusqlite::{params, Result};
use std::collections::HashMap;
use std::{collections::HashSet, path::PathBuf, time::Instant};
use tch::{CModule, Kind, Tensor};

pub fn create_database(
    connection_pool: Pool<SqliteConnectionManager>,
    path: PathBuf,
    vocab: &HashMap<String, usize>,
    model: &CModule,
) -> Result<(), String> {
    let batch_size: usize = get_create_batch_size();

    let path2 = path.clone();
    let start_time = Instant::now();

    let (tx, rx) = crossbeam_channel::unbounded();

    let existing_files_thread = std::thread::spawn({
        let connection_pool = connection_pool.clone();
        move || {
            let mut existing_files = HashSet::new();
            let connection = connection_pool
                .get()
                .expect("Unable to get connection from pool");
            let mut stmt = connection
                .prepare_cached("SELECT file_name, file_path FROM files")
                .expect("Failed to prepare statement.");
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0).expect("Problem with row_get"),
                        row.get::<_, String>(1).expect("Rows failed"),
                    ))
                })
                .expect("Failed to query result.");

            for row in rows {
                if let Ok((name, path)) = row {
                    existing_files.insert((name, path));
                }
            }
            existing_files
        }
    });

    let existing_files = existing_files_thread
        .join()
        .expect("Failed to join thread.");

    let conn = connection_pool
        .get()
        .expect("Could not get connection from pool");
    conn.execute_batch("PRAGMA journal_mode = WAL")
        .expect("Could not execute");

    let allowed_file_extensions = get_allowed_file_extensions().clone();
    let file_walking_thread = std::thread::spawn(move || {
        let mut batch: Vec<Files> = Vec::with_capacity(batch_size);
        WalkDir::new(&path)
            .follow_links(false)
            .into_iter()
            .for_each(|entry_result| {
                if let Ok(entry) = entry_result {
                    let path = entry.path();
                    if !should_ignore_path(&path)
                        && (path.is_dir() || is_allowed_file(&path, &allowed_file_extensions))
                    {
                        let path_slashes = convert_to_forward_slashes(&path);
                        let file = Files {
                            id: 0,
                            file_name: entry.file_name().to_string_lossy().into_owned(),
                            file_path: path_slashes,
                            file_type: if path.is_dir() {
                                Some("dir".to_string())
                            } else {
                                path.extension().and_then(|s| s.to_str()).map(String::from)
                            },
                        };
                        batch.push(file);
                        if batch.len() >= batch_size {
                            tx.send(std::mem::take(&mut batch))
                                .unwrap_or_else(|_| println!("Failed to send batch"));
                        }
                    }
                }
            });

        if !batch.is_empty() {
            tx.send(batch)
                .unwrap_or_else(|_| println!("Failed to send final batch"));
        }
    });
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

        if !batch_data.is_empty() {
            // Time database connection setup
            let mut connection = connection_pool
                .get()
                .expect("Unable to get connection from pool");
            let transaction = connection
                .transaction()
                .expect("Unable to create transaction");

            {
                // Time embedding generation
                let mut insert_stmt = transaction.prepare("INSERT INTO files (file_name, file_path, file_type, name_embeddings) VALUES (?, ?, ?, ?)")
                    .expect("Failed to prepare insertion file");

                let max_len = batch_data
                    .iter()
                    .map(|file_data| {
                        let tokens = tokenize_file_name(&file_data.1);
                        tokens_to_indices(tokens, &vocab).len()
                    })
                    .max()
                    .unwrap_or(0);

                let input_tensors: Vec<Tensor> = batch_data
                    .par_iter()
                    .map(|file_data| {
                        let tokens = tokenize_file_name(&file_data.1);
                        let mut token_indices: Vec<i64> = tokens_to_indices(tokens, &vocab)
                            .into_iter() // Convert Vec<usize> to iterator
                            .map(|x| x as i64) // Convert each usize to i64
                            .collect(); // Rebuild into Vec<i64>

                        // Pad with zeros to max_len
                        token_indices.resize(max_len, 0);

                        Tensor::from_slice(&token_indices)
                            .to_kind(Kind::Int64)
                            .unsqueeze(0)
                    })
                    .collect();

                let batch_tensor = Tensor::cat(&input_tensors, 0); // Concatenate along batch dimension
                let batch_embeddings = model
                    .method_ts("get_embedding", &[batch_tensor])
                    .expect("Batch embedding lookup failed");

                // Split batch results back into individual tensors if needed
                let embeddings: Vec<Tensor> = batch_embeddings
                    .chunk(input_tensors.len() as i64, 0)
                    .into_iter()
                    .collect();
                let embeddings_u8: Vec<Vec<u8>> = embeddings
                    .into_iter()
                    .map(|embedding| {
                        let numel = embedding.numel();
                        let mut embedding_vec_f32 = vec![0.0f32; numel];
                        embedding.copy_data(&mut embedding_vec_f32, numel);

                        embedding_vec_f32
                            .into_iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect()
                    })
                    .collect();

                // Time database insertion
                for (c, file_data) in batch_data.iter().enumerate() {
                    let file = &file_data.0;
                    if c < embeddings_u8.len() {
                        let vec = &embeddings_u8[c];
                        insert_stmt
                            .execute(params![
                                file.file_name,
                                file.file_path,
                                file.file_type.as_deref().map::<&str, _>(|s| s.as_ref()),
                                vec
                            ])
                            .expect("Could not insert file");
                    }
                }
            }
            // Time transaction commit
            transaction.commit().expect("Unable to commit transaction");
        }
    }
    file_walking_thread.join().expect("Failed to join thread.");

    println!(
        "Database population for {:?} took {}ms",
        path2,
        start_time.elapsed().as_millis()
    );
    Ok(())
}
