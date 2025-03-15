use crossbeam::channel;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use jwalk::WalkDir;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::prelude::*;
use rusqlite::{params, Result};
use std::sync::Arc;
use std::sync::Mutex;
use std::{collections::HashSet, path::PathBuf, time::Instant};

use crate::db_util::{convert_to_forward_slashes, is_allowed_file, should_ignore_path, Files};

pub fn create_database(
    connection_pool: Pool<SqliteConnectionManager>,
    path: PathBuf,
    allowed_file_extensions: &HashSet<String>,
    thread_pool: &rayon::ThreadPool,
) -> Result<(), String> {
    const BATCH_SIZE: usize = 250;

    println!(
        "Starting create_database function of Path {}",
        path.display()
    );
    let start_time = Instant::now();

    let (tx, rx) = channel::unbounded();
    let rx = Arc::new(Mutex::new(rx));

    println!("Threadpool {}", start_time.elapsed().as_millis());

    let model_thread = std::thread::spawn(|| {
        let model_time = Instant::now();
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::MultilingualE5Small).with_show_download_progress(true),
        )
        .expect("Could not create TextEmbedding");
        println!("Model took {}", model_time.elapsed().as_millis());
        model
    });

    let existing_files_thread = std::thread::spawn({
        let connection_pool = connection_pool.clone();
        move || {
            let existing_files_time = Instant::now();
            let mut existing_files = HashSet::new();
            let connection = connection_pool.get().unwrap();
            let mut stmt = connection
                .prepare_cached("SELECT file_name, file_path FROM files")
                .unwrap();
            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0).expect("Problem with row_get"),
                        row.get::<_, String>(1).expect("Rows failed"),
                    ))
                })
                .unwrap();

            for row in rows {
                if let Ok((name, path)) = row {
                    existing_files.insert((name, path));
                }
            }
            println!(
                "existing_files finished in {}ms",
                existing_files_time.elapsed().as_millis()
            );
            existing_files
        }
    });

    let model = model_thread.join().unwrap();
    let existing_files = existing_files_thread.join().unwrap();

    let conn = connection_pool
        .get()
        .expect("Could not get connection from pool");
    conn.execute_batch("PRAGMA journal_mode = WAL")
        .expect("Could not execute");

    let allowed_file_extensions = allowed_file_extensions.clone();
    let file_walking_thread = std::thread::spawn(move || {
        let tx = Arc::new(tx);
        let mut batch: Vec<Files> = Vec::with_capacity(BATCH_SIZE);
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
                        if batch.len() >= BATCH_SIZE {
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

    let rx = Arc::clone(&rx);
    while let Ok(batch) = rx.lock().unwrap().recv() {
        thread_pool.install(|| {
            // Prepare batch data
            let batch_data: Vec<_> = batch
                .par_iter()
                .filter_map(|file| {
                    if !existing_files.contains(&(file.file_name.clone(), file.file_path.clone())) {
                        let file_name_without_ext = file.file_name
                            .split_once('.')
                            .map(|(before, _)| before.to_string())
                            .unwrap_or(file.file_name.clone());
                        let file_name_finished = format!("query: {}", file_name_without_ext);
                        Some((file.clone(), file_name_finished))
                    } else {
                        None
                    }
                })
                .collect();

            if !batch_data.is_empty() {
                let vec_embed_time = Instant::now();

                // Prepare names for batch embedding
                let names_to_embed: Vec<String> = batch_data.iter().map(|(_, name)| name.clone()).collect();

                // Perform batch embedding
                let name_embeddings = model.embed(names_to_embed, Some(BATCH_SIZE)).expect("Could not embed files");

                // Process embeddings and insert into database
                let mut connection = connection_pool.get().unwrap();
                let transaction = connection.transaction().unwrap();
                {
                    let mut insert_stmt = transaction.prepare("INSERT INTO files (file_name, file_path, file_type, name_embeddings) VALUES (?, ?, ?, ?)").expect("Failed to prepare insertion file");

                    for ((file, _), embedding) in batch_data.iter().zip(name_embeddings.iter()) {
                        let slice_f32 = embedding.as_slice();
                        let name_vec_embedded: Vec<u8> = unsafe { std::mem::transmute(slice_f32.to_vec()) };

                        insert_stmt.execute(params![
                        file.file_name,
                        file.file_path,
                        file.file_type.as_deref().map::<&str, _>(|s| s.as_ref()),
                        name_vec_embedded
                    ]).expect("Could not insert file");
                    }
                }
                transaction.commit().unwrap();

                println!("Vec embed took {} ms", vec_embed_time.elapsed().as_millis());
            }
        });
    }

    file_walking_thread.join().unwrap();

    println!("{}", start_time.elapsed().as_millis());
    Ok(())
}
