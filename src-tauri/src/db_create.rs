use crate::config_handler::{
    get_allowed_file_extensions, get_create_batch_size, get_embedding_dimensions,
};
use crate::db_util::{convert_to_forward_slashes, full_emb, is_allowed_file, Files};
use jwalk::WalkDir;
use ndarray::Array2;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rusqlite::{params, Result};
use std::{collections::HashSet, path::PathBuf, time::Instant};

/// This Function takes in a connection pool as well as a Path as Input
/// and then recursivly checks for every file / dir from the Path and adds it to the database.
/// The Vocabulary of the skip-gram model as well as the weights are being used via the pub static Oncelocks.
pub fn create_database(
    connection_pool: Pool<SqliteConnectionManager>,
    path: PathBuf,
) -> Result<(), String> {
    let batch_size: usize = get_create_batch_size();

    let path2 = path.clone();
    let start_time = Instant::now();

    // Starting Channel
    let (tx, rx) = crossbeam_channel::bounded(batch_size * 5);

    // Creates a Thread that genrates Hashset of every Name as well as Path from the Database
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

            for (name, path) in rows.flatten() {
                existing_files.insert((name, path));
            }
            existing_files
        }
    });

    //closing Thread and getting Values
    let existing_files: HashSet<(String, String)> = existing_files_thread
        .join()
        .expect("Failed to join thread.");

    let conn = connection_pool
        .get()
        .expect("Could not get connection from pool");

    // Activating Write Ahead Logging which enables reading and writing at the same time, it should theoretically already be enabled but to be safe
    conn.execute_batch("PRAGMA journal_mode = WAL")
        .expect("Could not enable WAL");

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
                                tx.send(std::mem::take(&mut batch))
                                    .unwrap_or_else(|_| println!("Failed to send batch"));
                            }
                        } else {
                            println!(
                                "Warning: Couldn't get file stem for {:?}, skipping entry.",
                                entry.path()
                            );
                        }
                    }
                }
            });

        // Sends the last Batch
        if !batch.is_empty() {
            tx.send(batch)
                .unwrap_or_else(|_| println!("Failed to send final batch"));
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

        //If there is Batch Data start Processing and get Connections and prepare SQL Statement
        if !batch_data.is_empty() {
            let mut connection = connection_pool
                .get()
                .expect("Unable to get connection from pool");
            let transaction = connection
                .transaction()
                .expect("Unable to create transaction");

            {
                //Preparing SQL Statement for Inserting Data into the DB
                let mut insert_stmt = transaction.prepare("INSERT INTO files (file_name, file_path, file_type, name_embeddings) VALUES (?, ?, ?, ?)")
                    .expect("Failed to prepare insertion file");

                //The Embedding takes up like 80% of the time per Batch

                //Embeds the Batch and writes it as a Matrix
                let batch_embeddings: Array2<f32> = {
                    let embeddings: Vec<Vec<f32>> = batch_data
                        .par_iter()
                        .map(|file_data| {
                            let file_name = &file_data.1;
                            full_emb(file_name)
                        })
                        .collect();

                    let n_samples = embeddings.len();
                    Array2::from_shape_vec(
                        (n_samples, get_embedding_dimensions()),
                        embeddings.into_iter().flatten().collect(),
                    )
                    .expect("Shape mismatch")
                };

                // Transform the every Embedding into a Vec<u8> so that they can be stored in the Database
                let embeddings_u8: Vec<Vec<u8>> = batch_embeddings
                    .outer_iter()
                    .map(|embedding_row| {
                        embedding_row.iter().flat_map(|f| f.to_le_bytes()).collect()
                    })
                    .collect();

                //c is a Counter that is used to check if there is a mismatch between embeddings_u8 and the Rest of the Dat
                //The Batch Data is inserted into the Database
                for (c, file_data) in batch_data.iter().enumerate() {
                    let file = &file_data.0;
                    if c < embeddings_u8.len() {
                        let vec = &embeddings_u8[c];
                        insert_stmt
                            .execute(params![file.file_name, file.file_path, file.file_type, vec])
                            .unwrap_or_else(|_| panic!("Could not insert file {:?}", file));
                    }
                }
            }
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
