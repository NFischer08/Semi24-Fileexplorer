use crate::config_handler::ALLOWED_FILE_EXTENSIONS;
use jwalk::WalkDir;
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use rayon::iter::split;
use rayon::prelude::*;
use rusqlite::{params, MappedRows, Result, Row};
use std::cmp::Ordering;
use std::fmt::format;
use std::sync::mpsc;
use std::{
    collections::HashSet,
    fs::{self, DirEntry},
    path::{Path, PathBuf},
    time::Instant,
};
use strsim::normalized_levenshtein;

#[derive(Debug)]
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
            file_type TEXT NOT NULL
    )",
        (),
    )?;

    pooled_connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_file_path ON files (file_path)",
        [],
    )?;
    let x = if let Some(allowed_ext) = ALLOWED_FILE_EXTENSIONS.get().clone() {
        Ok(allowed_ext.clone())
    } else {
        Ok(HashSet::new())
    };
    println!("DB: allowed file extensions: {:?}", x);
    x
}

pub fn create_database(
    mut pooled_connection: PooledConnection<SqliteConnectionManager>,
    path: PathBuf,
    allowed_file_extensions: &HashSet<String>,
    thread_pool: &rayon::ThreadPool,
) -> Result<(), String> {
    println!(
        "Starting create_database function of Path {}",
        path.display()
    );

    let start_time = Instant::now();
    let files_vec: Vec<Files> = thread_pool.install(|| {
        WalkDir::new(&path)
            .into_iter()
            .par_bridge()
            .filter_map(|entry_result| {
                entry_result.ok().and_then(|entry| {
                    let path = entry.path();
                    if !should_ignore_path(&path)
                        && (path.is_dir() || is_allowed_file(&path, allowed_file_extensions))
                    {
                        let path_slashes = convert_to_forward_slashes(&path);
                        Some(Files {
                            id: 0,
                            file_name: entry.file_name().to_string_lossy().into_owned(),
                            file_path: path_slashes,
                            file_type: if path.is_dir() {
                                Some("dir".to_string())
                            } else {
                                path.extension().and_then(|s| s.to_str()).map(String::from)
                            },
                        })
                    } else {
                        None
                    }
                })
            })
            .collect()
    });
    println!(
        "Directory scan completed in {:?} of Path {}",
        start_time.elapsed(),
        path.display()
    );

    let tx = match pooled_connection.transaction() {
        Ok(tx) => tx,
        Err(e) => return Err(e.to_string()),
    };
    {
        let mut existing_files = HashSet::new();
        let mut stmt = match tx.prepare_cached("SELECT file_name, file_path FROM files") {
            Ok(stmt) => stmt,
            Err(e) => return Err(e.to_string()),
        };
        let rows = match stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }) {
            Ok(rows) => rows,
            Err(e) => return Err(e.to_string()),
        };

        for row in rows {
            if let Ok((name, path)) = row {
                existing_files.insert((name, path));
            }
        }

        let mut insert_stmt = match tx
            .prepare_cached("INSERT INTO files (file_name, file_path, file_type) VALUES (?, ?, ?)")
        {
            Ok(stmt) => stmt,
            Err(e) => return Err(e.to_string()),
        };
        let batch_size = 100000;
        let mut inserted_count = 0;
        for (i, chunk) in files_vec.chunks(batch_size).enumerate() {
            for file in chunk {
                if !existing_files
                    .contains(&(file.file_name.to_string(), file.file_path.to_string()))
                {
                    match insert_stmt.execute(params![
                        file.file_name,
                        file.file_path,
                        file.file_type.as_deref().map::<&str, _>(|s| s.as_ref())
                    ]) {
                        Ok(_) => (),
                        Err(e) => return Err(e.to_string()),
                    };
                    inserted_count += 1;
                }
            }
            if (i + 1) % 10 == 0 {
                // Progress reporting can be added here if needed
            }
        }
        println!(
            "Total files inserted: {} of Path {}",
            inserted_count,
            path.display()
        );
    }
    match tx.commit() {
        Ok(_) => {}
        Err(e) => return Err(e.to_string()),
    };

    println!(
        "create_database function completed in {:?} of Path {}",
        start_time.elapsed(),
        path.display()
    );
    Ok(())
}

pub fn check_database(
    mut conn: PooledConnection<SqliteConnectionManager>,
    allowed_file_extensions: &HashSet<String>,
    thread_pool: &rayon::ThreadPool,
) -> Result<(), Box<dyn std::error::Error>> {
    let bad_paths: Vec<String> = {
        let mut stmt = conn.prepare_cached("SELECT file_path FROM files")?;
        let rows: Vec<Result<String, _>> =
            stmt.query_map([], |row| row.get::<_, String>(0))?.collect();

        thread_pool.install(|| {
            rows.into_par_iter()
                .filter_map(|path_result| {
                    path_result.ok().and_then(|path| {
                        let path_obj = Path::new(&path);
                        if !is_allowed_file(path_obj, &allowed_file_extensions)
                            && !fs::metadata(&path).is_ok()
                        {
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
    path.to_str()
        .map_or(false, |s| s.starts_with("/proc") || s.starts_with("/sys"))
}

pub fn search_database(
    connection_pool: Pool<SqliteConnectionManager>,
    search_term: &str,
    similarity_threshold: f64,
    thread_pool: &rayon::ThreadPool,
    search_path: PathBuf,
    search_file_type: &str,
) -> Result<Vec<DirEntry>> {
    let conn = connection_pool.get().expect("get connection pool");
    const BATCH_SIZE: usize = 1000; // Adjust this value as needed

    let start_time = Instant::now();

    // Convert searchpath to a string
    let search_path_str = if cfg!(windows) && search_path.to_str().unwrap_or("") == "/" {
        String::new()
    } else {
        search_path.to_str().unwrap_or("").to_string()
    };

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_file_path ON files(file_path)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_file_type ON files(file_type)",
        [],
    )?;

    let search_file_type = search_file_type.replace(" ", "");

    println!("query Results {}", start_time.elapsed().as_millis());

    /*
    println!("search_path_str: '{}'", search_path_str);
    println!("search_file_type: '{}'", search_file_type);
     */

    let (tx, rx) = mpsc::channel();
    let query_thread = std::thread::spawn(move || {
        let conn = connection_pool.get().expect("get connection pool");

        let mut stmt = conn.prepare("SELECT file_path FROM files WHERE  file_path LIKE ?1 AND (CASE WHEN ?2 = '' THEN 1 ELSE (',' || ?2 || ',') LIKE ('%,' || file_type || ',%') END)").map_err(|e| {
            eprintln!("Failed to prepare statement: {:?}", e);
            e
        })?;

        let query_result = stmt.query_map(
            params![format!("{}%", search_path_str), search_file_type],
            |row| Ok(row.get::<_, String>(0)?),
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

    let mut results: Vec<(String, f64)> = thread_pool.install(|| {
        rx.into_iter()
            .flat_map(|vec_of_paths| vec_of_paths)
            .par_bridge() // Convert to parallel iterator
            .filter_map(|file_path| {
                let file_name = Path::new(&file_path).file_stem()?.to_str()?;
                let similarity = normalized_levenshtein(file_name, &search_term);
                if similarity >= similarity_threshold {
                    println!("{}", file_path);
                    Some((file_path, similarity))
                } else {
                    None
                }
            })
            .collect()
    });

    query_thread.join().expect("Query thread panicked")?;

    println!("results took: {}", start_time.elapsed().as_millis());

    results.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let return_entries: Vec<DirEntry> = results
        .into_iter()
        .filter_map(|(s, _)| {
            let path = Path::new(&s);
            fs::read_dir(path.parent().unwrap_or(Path::new(".")))
                .ok()
                .and_then(|mut dir| {
                    dir.find(|e| e.as_ref().map(|d| d.path() == path).unwrap_or(false))
                })
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
