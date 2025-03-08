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

#[derive(Debug, Clone)]
struct Files {
    id: i32,
    file_name: String,
    file_path: String,
    file_type: Option<String>,
}

pub fn initialize_database_and_extensions(
    pooled_connection: &PooledConnection<SqliteConnectionManager>,
) -> Result<HashSet<String>, Box<dyn std::error::Error>> {

    pooled_connection.execute(
        "CREATE TABLE IF NOT EXISTS files (
            id   INTEGER PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            name_embeddings BLOB NOT NULL
    )",
        (),
    )?;

    pooled_connection.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON files (file_path)", [])?;

    // Get allowed file extensions
    let allowed_file_extensions: HashSet<String> = [
        // Text and documents
        "txt", "pdf", "doc", "docx", "rtf", "odt", "tex", "md", "epub",
        // Spreadsheets and presentations
        "xls", "xlsx", "csv", "ods", "ppt", "pptx", "odp", "key",
        // Images
        "jpg", "jpeg", "png", "gif", "bmp", "tiff", "svg", "webp", "ico", "raw",
        // Audio and video
        "mp3", "wav", "ogg", "flac", "aac", "wma", "m4a", "mp4", "avi", "mov", "wmv", "flv", "mkv", "webm", "m4v", "3gp",
        // Archives and data
        "zip", "rar", "7z", "tar", "gz", "bz2", "xz", "json", "xml", "yaml", "yml", "toml", "ini", "cfg",
        // Web and programming
        "html", "htm", "css", "js", "php", "asp", "jsp", "py", "java", "c", "cpp", "h", "hpp", "cs", "rs", "go", "rb", "pl", "swift", "kt", "ts", "coffee", "scala", "groovy", "lua", "r",
        // Scripts and executables
        "sh", "bash", "zsh", "fish", "bat", "cmd", "ps1", "exe", "dll", "so", "dylib",
        // Other formats
        "sql", "db", "sqlite", "mdb", "ttf", "otf", "woff", "woff2", "obj", "stl", "fbx", "dxf", "dwg", "psd", "ai", "ind", "iso", "img", "dmg", "bak", "tmp", "log", "pcap"
    ].iter().map(|&s| String::from(s)).collect();

    Ok(allowed_file_extensions)
}


pub fn create_database(
    mut connection_pool: Pool<SqliteConnectionManager>,
    path: PathBuf,
    allowed_file_extensions: &HashSet<String>,
    thread_pool: &rayon::ThreadPool,
) -> Result<(), String> {
    const BATCH_SIZE: usize = 100;

    println!("Starting create_database function of Path {}", path.display());

    let start_time = Instant::now();

    let (tx, rx) = mpsc::channel();

    println!("Threadpool {}", start_time.elapsed().as_millis());

        thread_pool.install(|| {
            let tx = Arc::new(tx); // Wrap the sender in an Arc

            let mut batch: Vec<Files> = Vec::new();

            WalkDir::new(&path).follow_links(false)
                .into_iter()
                .par_bridge()
                .for_each_with(Vec::with_capacity(BATCH_SIZE), |batch, entry_result| {
                    if let Ok(entry) = entry_result {
                        let path = entry.path();
                        if !should_ignore_path(&path) && (path.is_dir() || is_allowed_file(&path, allowed_file_extensions)) {
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
                                let tx_clone = Arc::clone(&tx);
                                tx_clone.send(batch.clone()).unwrap_or_else(|_| println!("Failed to send batch"));
                                batch.clear();
                            }
                        }
                    }
                });

            if !batch.is_empty() {
                let tx_clone = Arc::clone(&tx);
                tx_clone.send(batch.clone()).unwrap_or_else(|_| println!("Failed to send final batch"));
            }
            drop(tx);
        });
    println!("Directory scan completed in {:?} of Path {}", start_time.elapsed(), path.display());


        let model_time = Instant::now();
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::MultilingualE5Small)
                .with_show_download_progress(true)
        ).expect("Could not create TextEmbedding");


        println!("Model took {}", model_time.elapsed().as_millis());

    let mut existing_files = HashSet::new();

    let connection = connection_pool.get().unwrap();

    let mut stmt = match connection.prepare_cached("SELECT file_name, file_path FROM files") {
        Ok(stmt) => stmt,
        Err(e) => return Err(e.to_string())
    };
    let rows = match stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1).expect("Could not parse row")))
    }) {
        Ok(rows) => rows,
        Err(e) => return Err(e.to_string())
    };

    for row in rows {
        if let Ok((name, path)) = row {
            existing_files.insert((name, path));
        }
    }

    let _ = thread_pool.install(move || -> Result<(), String> {
        println!("Threadpool starting");
        let conn = connection_pool.get().expect("Could not get connection from pool");
        conn.execute_batch("PRAGMA journal_mode = WAL").expect("Could not execute");

        let mut insert_stmt = conn.prepare("
        INSERT INTO files (file_name, file_path, file_type, name_embeddings)
        VALUES (?, ?, ?, ?)
    ").map_err(|e| e.to_string()).expect("Could not insert file");

        for batch in rx {
            println!("Batch received {}", batch.len());
            for file in batch {
                if !existing_files.contains(&(file.file_name.clone(), file.file_path.clone())) {
                    let file_name_without_ext = file.file_name.split_once('.').map(|(before, _)| before.to_string()).unwrap_or(file.file_name.clone());
                    let mut name_vec = Vec::new();
                    let file_name_finished = "query: ".to_string() + &file_name_without_ext;
                    name_vec.push(file_name_finished);
                    let name_embedding = model.embed(name_vec, None).expect("Could not embed file");
                    let slice_f32 = name_embedding[0].as_slice();
                    let name_vec_embedded: Vec<u8> = unsafe { std::mem::transmute(slice_f32.to_vec()) };

                    insert_stmt.execute(params![
                    file.file_name,
                    file.file_path,
                    file.file_type.as_deref().map::<&str, _>(|s| s.as_ref()),
                    name_vec_embedded
                ]).expect("Could not insert file");
                }
            }
            connection_pool.get().unwrap().transaction().expect("Transaction failed").commit().expect("Commit transaction failed");
        }
        Ok(())
    });

    println!("Path {}, time taken {}", path.display(), start_time.elapsed().as_millis());
    println!("create_database function completed in {:?} of Path {}", start_time.elapsed(), path.display());
    Ok(())
    }


pub fn check_database(
    mut conn: PooledConnection<SqliteConnectionManager>,
    allowed_file_extensions: &HashSet<String>,
    thread_pool: &rayon::ThreadPool,
) -> Result<(), Box<dyn std::error::Error>> {
    let bad_paths: Vec<String> = {
        let mut stmt = conn.prepare_cached("SELECT file_path FROM files")?;
        let rows: Vec<Result<String, _>> = stmt.query_map([], |row| row.get::<_, String>(0))?.collect();

        thread_pool.install(|| {
            rows.into_par_iter()
                .filter_map(|path_result| {
                    path_result.ok().and_then(|path| {
                        let path_obj = Path::new(&path);
                        if !is_allowed_file(path_obj, &allowed_file_extensions) &&
                            !fs::metadata(&path).is_ok() {
                            Some(path)
                        } else {
                            None
                        }
                    })
                })
                .collect()
        })
    };

    println!("Number of bad files: {}", bad_paths.len());

    if !bad_paths.is_empty() {
        let tx = conn.transaction()?;
        {
            let placeholders = bad_paths.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let query = format!("DELETE FROM files WHERE file_path IN ({})", placeholders);
            let mut stmt = tx.prepare_cached(&query)?;
            stmt.execute(rusqlite::params_from_iter(bad_paths.iter()))?;
        }
        tx.commit()?;
    }
    println!("Check Database completed");

    Ok(())
}

fn is_allowed_file(path: &Path, allowed_file_extensions: &HashSet<String>) -> bool {
    if should_ignore_path(path) {
        return false;
    }
    path.extension()
        .and_then(|s| s.to_str())
        .map(|ext| allowed_file_extensions.contains(ext))
        .unwrap_or(false)
}


fn should_ignore_path(path: &Path) -> bool {
    path.to_str().map_or(false, |s| s.starts_with("/proc") || s.starts_with("/sys"))
}

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

    pooled_connection.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON files(file_path)", [])?;
    pooled_connection.execute("CREATE INDEX IF NOT EXISTS idx_file_type ON files(file_type)", [])?;

    let search_file_type = search_file_type.replace(" " ,"");

    println!("query Results {}", start_time.elapsed().as_millis());

    /*
    println!("search_path_str: '{}'", search_path_str);
    println!("search_file_type: '{}'", search_file_type);
     */

    let (tx, rx) = mpsc::channel();
    let query_thread = std::thread::spawn(move || {

        let pooled_connection = connection_pool.get().expect("get connection pool");

        let mut stmt = pooled_connection.prepare("SELECT file_path, name_embeddings FROM files WHERE  file_path LIKE ?1 AND (CASE WHEN ?2 = '' THEN 1 ELSE (',' || ?2 || ',') LIKE ('%,' || file_type || ',%') END)").map_err(|e| {
            eprintln!("Failed to prepare statement: {:?}", e);
            e
        })?;

        let query_result = stmt.query_map(
            params![
        format!("{}%", search_path_str),
        search_file_type
    ],
            |row| Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Vec<u8>>(1)?
            ))
        )?;

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
                if similarity > 0.8 {
                    println!("similarity = {}, file_name: {}", similarity, row.0);
                    Some((row.0, similarity.cast()))
                } else {
                    None
                }
                /*
                }
                let file_name = Path::new(&row.0).file_name()?.to_str()?;
                let similarity = normalized_levenshtein(file_name, &search_term);
                if similarity >= similarity_threshold {
                    Some((row.0, similarity))
                } else {
                    None
                }

                 */
            })
            .collect()
    });

    query_thread.join().expect("Query thread panicked")?;

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

fn convert_to_forward_slashes(path: &PathBuf) -> String {
    path.to_str()
        .map(|s| s.replace('\\', "/"))
        .unwrap_or_else(|| String::new())
}

fn cosine_similarity(
    search_embedding: &Vec<f32>,
    candidate_embedding: &Vec<f32>) -> f32 {
    let mut a2: f32 = 0.0;
    let mut b2: f32 = 0.0;
    let mut ab: f32 = 0.0;

    for (a ,b) in search_embedding.iter().zip(candidate_embedding.iter()) {
        a2 +=a *a;
        b2 += b * b;
        ab += a*b;
    }
    let result = ab/a2.sqrt()/b2.sqrt();
    result
}