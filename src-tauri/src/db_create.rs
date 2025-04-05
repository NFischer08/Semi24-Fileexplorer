use fastembed::{TextEmbedding};
use jwalk::WalkDir;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::prelude::*;
use rusqlite::{params, Result};
use std::{collections::HashSet, path::PathBuf, thread, time::Instant};
use once_cell::sync::Lazy;
use tch::{CModule, Tensor, Kind};
use crate::db_util::{convert_to_forward_slashes, is_allowed_file, should_ignore_path, Files, tokenize_file_name, load_vocab, tokens_to_indices};

pub fn create_database(
    connection_pool: Pool<SqliteConnectionManager>,
    path: PathBuf,
    allowed_file_extensions: &HashSet<String>,
    thread_pool: &rayon::ThreadPool,
    model: &Lazy<TextEmbedding>,
    pymodel_path: &str
) -> Result<(), String> {

    tch::set_num_threads(num_cpus::get() as i32);

    const BATCH_SIZE: usize = 250;

    println!(
        "Starting create_database function of Path {}",
        path.display()
    );
    let start_time = Instant::now();

    let (tx, rx) = crossbeam_channel::unbounded();

    println!("Threadpool {}", start_time.elapsed().as_millis());

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

    let existing_files = existing_files_thread.join().unwrap();

    let conn = connection_pool
        .get()
        .expect("Could not get connection from pool");
    conn.execute_batch("PRAGMA journal_mode = WAL")
        .expect("Could not execute");

    let allowed_file_extensions = allowed_file_extensions.clone();
    let file_walking_thread = std::thread::spawn(move || {
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
                            //println!("Batch send");
                            tx.send(std::mem::take(&mut batch))
                                .unwrap_or_else(|_| println!("Failed to send batch"));
                        }
                    }
                }
            });

        if !batch.is_empty() {
            println!("Last Batch send");
            tx.send(batch)
                .unwrap_or_else(|_| println!("Failed to send final batch"));
        }
    });

    thread_pool.install(move || {
        println!("Starting thread pool");
        while let Ok(batch) = rx.recv() {
            // Time batch preparation
            let batch_data: Vec<_> = batch
                .iter()
                .filter_map(|file| {
                    if !existing_files.contains(&(file.file_name.clone(), file.file_path.clone())) {
                        let file_name_without_ext = file.file_name
                            .split_once('.')
                            .map(|(before, _)| before.to_string())
                            .unwrap_or(file.file_name.clone());
                        let file_name_finished = file_name_without_ext;
                        Some((file.clone(), file_name_finished))
                    } else {
                        None
                    }
                })
                .collect();

            if !batch_data.is_empty() {
                // Time database connection setup
                let db_start = Instant::now();
                let mut connection = connection_pool.get().unwrap();
                let transaction = connection.transaction().unwrap();
                println!("🕒 DB connection setup took: {:?}", db_start.elapsed());

                {
                    // Time embedding generation
                    let embeddings_start = Instant::now();
                    let mut insert_stmt = transaction.prepare("INSERT INTO files (file_name, file_path, file_type, name_embeddings) VALUES (?, ?, ?, ?)")
                        .expect("Failed to prepare insertion file");

                    let vocab = load_vocab("src-tauri/src/neural_network/vocab.json");
                    let model = CModule::load(pymodel_path).expect("Failed to load model");

                    let embeddings: Vec<Vec<u8>> = batch_data.par_iter()
                        .map(|file_data| {

                            println!("Running on thread: {:?}", thread::current().id());

                            let file_name = &file_data.1;
                            let tokens = tokenize_file_name(file_name);
                            let token_indices = tokens_to_indices(tokens, &vocab);

                            let tokens_indices_i64: Vec<i64> = token_indices.iter()
                                .map(|&x| x as i64)
                                .collect();

                            let input_tensor = Tensor::from_slice(&tokens_indices_i64)
                                .to_kind(Kind::Int64)
                                .unsqueeze(0);

                            let embedding = model.method_ts("get_embedding", &[input_tensor]).expect("Inference failed");

                            let numel = embedding.numel();

                            let mut embedding_vec_f32 = vec![0.0f32; numel];
                            embedding.copy_data(&mut embedding_vec_f32, numel);

                            let result = embedding_vec_f32.into_iter()
                                .flat_map(|f| f.to_le_bytes())
                                .collect();

                            result

                        })
                        .collect();
                    println!("🕒 Embeddings generation took: {:?}", embeddings_start.elapsed());

                    // Time database insertion
                    let insert_start = Instant::now();
                    for (c, file_data) in batch_data.iter().enumerate() {
                        let file = &file_data.0;
                        if c < embeddings.len() {
                            let vec = &embeddings[c];
                            insert_stmt.execute(params![
                        file.file_name,
                        file.file_path,
                        file.file_type.as_deref().map::<&str, _>(|s| s.as_ref()),
                        vec
                    ]).expect("Could not insert file");
                        }
                    }
                    println!("🕒 Database insertion took: {:?}", insert_start.elapsed());
                }

                // Time transaction commit
                let commit_start = Instant::now();
                transaction.commit().unwrap();
                println!("🕒 Transaction commit took: {:?}", commit_start.elapsed());
            }
        }
    });

    file_walking_thread.join().unwrap();

    println!("{}", start_time.elapsed().as_millis());
    Ok(())
}
