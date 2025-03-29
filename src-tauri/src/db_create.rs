use crossbeam::channel;
use fastembed::{TextEmbedding};
use jwalk::WalkDir;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::prelude::*;
use rusqlite::{params, Result};
use std::sync::Arc;
use std::sync::Mutex;
use std::{collections::HashSet, fs, path::PathBuf, time::Instant};
use std::collections::HashMap;
use once_cell::sync::Lazy;
use tch::{CModule, Tensor, Kind};
use tch::nn::Module;
use crate::db_util::{convert_to_forward_slashes, is_allowed_file, should_ignore_path, Files};

pub fn create_database(
    connection_pool: Pool<SqliteConnectionManager>,
    path: PathBuf,
    allowed_file_extensions: &HashSet<String>,
    thread_pool: &rayon::ThreadPool,
    model: &Lazy<TextEmbedding>,
    pymodel : &Lazy<CModule, fn() -> CModule>
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
                        //Needed for Fastembed let file_name_finished = format!("query: {}", file_name_without_ext);
                        let file_name_finished = file_name_without_ext;
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

                let tokenized_names: Vec<Vec<&str>> = names_to_embed
                    .iter()
                    .map(|name| name.split_whitespace().collect())
                    .collect();

                let padded_tokenized_names: Vec<Vec<&str>> = tokenized_names
                    .iter()
                    .map(|tokens| {
                        let mut padded = tokens.clone(); // Clone the current vector
                        padded.resize(10, ""); // Resize to 10, padding with empty strings
                        padded
                    })
                    .collect();

                let vocab_path = "src-tauri/src/neural_network/vocab.json";
                let vocab: HashMap<String, i64> = serde_json::from_str(&fs::read_to_string(vocab_path).unwrap())
                    .expect("Failed to load vocabulary");

                let indexed_names: Vec<Vec<i64>> = padded_tokenized_names
                    .iter()
                    .map(|tokens| {
                        tokens
                            .iter()
                            .map(|token| *vocab.get(*token).unwrap_or(&0)) // Default to 0 for unknown tokens
                            .collect()
                    })
                    .collect();

                let tensors: Vec<Tensor> = indexed_names
                    .iter()
                    .map(|indices| Tensor::from_slice(indices).to_kind(Kind::Int64))
                    .collect();

                for (i, tensor) in tensors.iter().enumerate() {
                    println!("Tensor {} shape: {:?}", i, tensor.size());
                }

                let batch_tensor = Tensor::stack(&tensors, 0); // Stack along dimension 0

                let pairs_file = fs::read_to_string("src-tauri/src/neural_network/skipgram_pairs.json").expect("Failed to read file");
                let data: Vec<(i64, i64)> = serde_json::from_str(&pairs_file).expect("Failed to parse JSON");

                let center_word_indices: Vec<i64> = data.iter().map(|(center, _)| *center).collect();
                let context_word_indices: Vec<i64> = data.iter().map(|(_, context)| *context).collect();

                let center_tensor = Tensor::from_slice(&center_word_indices);
                let context_tensor = Tensor::from_slice(&context_word_indices);

                //let embeddings = pymodel.forward(&center_tensor, &context_tensor);

                //let embeddings = pymodel.forward(&batch_tensor);

                let embeddings = pymodel
                    .forward_ts(&[center_tensor, context_tensor])
                    .expect("Failed to execute forward pass");

                let aggregated_embeddings: Tensor = embeddings.sum_dim_intlist([0].as_ref(), true, Kind::Float);

                let embedding_vecs: Vec<Vec<f32>> = aggregated_embeddings
                    .split(1, 0) // Split along batch dimension
                    .iter()
                    .map(|tensor| {
                        let vec: Vec<f32> = tensor.try_into().expect("Failed to convert tensor to Vec<f32>");
                        vec
                    })
                    .collect();


                let embedding_vecs_u8: Vec<Vec<u8>> = embedding_vecs
                    .iter()
                    .map(|inner_vec| {
                        inner_vec
                            .iter()
                            .map(|&value| {
                                // Clamp the value to the range of u8 and convert
                                value.round().clamp(0.0, 255.0) as u8
                            })
                            .collect()
                    })
                    .collect();

                let mut connection = connection_pool.get().unwrap();
                let transaction = connection.transaction().unwrap();
                {
                    let mut insert_stmt = transaction.prepare("INSERT INTO files (file_name, file_path, file_type, name_embeddings) VALUES (?, ?, ?, ?)").expect("Failed to prepare insertion file");

                    for ((file, _), embedding) in batch_data.iter().zip(embedding_vecs_u8.iter())
                    {
                        insert_stmt.execute(params![
                        file.file_name,
                        file.file_path,
                        file.file_type.as_deref().map::<&str, _>(|s| s.as_ref()),
                        embedding
                    ]).expect("Failed to insert embedding");
                    }
                }
                transaction.commit().unwrap();

                /*

                println!("names_to_embed {}", names_to_embed.join("\n"));


                // Perform batch embedding
                let name_embeddings = model.embed(names_to_embed, Some(BATCH_SIZE)).expect("Could not embed files");

                // Process embeddings and insert into database
                let mut connection = connection_pool.get().unwrap();
                let transaction = connection.transaction().unwrap();
                {
                    let mut insert_stmt = transaction.prepare("INSERT INTO files (file_name, file_path, file_type, name_embeddings) VALUES (?, ?, ?, ?)").expect("Failed to prepare insertion file");

                    for ((file, _), embedding) in batch_data.iter().zip(name_embeddings.iter()) {
                        let slice_f32 = embedding.as_slice();
                        let name_vec_embedded: Vec<u8> = slice_f32.to_vec().into_iter()
                            .flat_map(|f| f.to_le_bytes())
                            .collect();

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

                 */
            }
        });
    }

    file_walking_thread.join().unwrap();

    println!("{}", start_time.elapsed().as_millis());
    Ok(())
}
