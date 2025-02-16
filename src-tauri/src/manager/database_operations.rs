use rusqlite::{params, Result};
use rayon::prelude::*;
use jwalk::WalkDir;
use r2d2::{PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::time::Instant;
use strsim::normalized_levenshtein;
use std::fs::DirEntry;
use std::fs;
#[derive(Debug)]
struct Files {
    id: i32,
    file_name: String,
    file_path: String,
    file_type: Option<String>,
}

pub fn initialize_database_and_extensions(
    connection: &PooledConnection<SqliteConnectionManager>,
) -> Result<HashSet<String>, Box<dyn std::error::Error>> {

    connection.execute(
        "CREATE TABLE IF NOT EXISTS files (
        id   INTEGER PRIMARY KEY,
        file_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_type  BLOB
    )",
        (),
    )?;

    connection.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON files (file_path)", [])?;

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
    conn: PooledConnection<SqliteConnectionManager>,
    path: PathBuf,
    allowed_file_extensions: &HashSet<String>,
    thread_pool: &rayon::ThreadPool,
) -> Result<(), String> {
    println!("Starting create_database function");
    println!("Scanning directory: {}", path.display());


    let start_time = Instant::now();
    let files_vec: Vec<Files> = thread_pool.install(|| {
        WalkDir::new(&path)
            .parallelism(jwalk::Parallelism::RayonNewPool(num_cpus::get()))
            .into_iter()
            .par_bridge()
            .filter_map(|entry_result| {
                entry_result.ok().and_then(|entry| {
                    let path = entry.path();
                    if !should_ignore_path(&path) && (path.is_dir() || is_allowed_file(&path, allowed_file_extensions)) {
                        let path_slashes = convert_to_forward_slashes(&path);
                        Some(Files {
                            id: 0,
                            file_name: entry.file_name().to_string_lossy().into_owned(),
                            file_path: path_slashes,
                            file_type: if path.is_dir() {
                                Some("directory".to_string())
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
    println!("Directory scan completed in {:?}", start_time.elapsed());
    println!("Number of files to insert: {}", files_vec.len());

    let mut conn = conn;
    let tx = match conn.transaction() {
        Ok(tx) => tx,
        Err(e) => return Err(e.to_string())
    };
    {
        println!("Starting to fetch existing files");
        let fetch_start = Instant::now();
        let mut existing_files = HashSet::new();
        let mut stmt = match tx.prepare_cached("SELECT file_name, file_path FROM files") {
            Ok(stmt) => stmt,
            Err(e) => return Err(e.to_string())
        };
        let rows = match stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        }) {
            Ok(rows) => rows,
            Err(e) => return Err(e.to_string())
        };

        for row in rows {
            if let Ok((name, path)) = row {
                existing_files.insert((name, path));
            }
        }
        println!("Fetched {} existing files in {:?}", existing_files.len(), fetch_start.elapsed());

        println!("Starting file insertion");
        let insert_start = Instant::now();
        let mut insert_stmt = match tx.prepare_cached("INSERT INTO files (file_name, file_path, file_type) VALUES (?, ?, ?)") {
            Ok(stmt) => stmt,
            Err(e) => return Err(e.to_string())
        };
        let batch_size = 100000;
        let mut inserted_count = 0;
        for (i, chunk) in files_vec.chunks(batch_size).enumerate() {
            for file in chunk {
                if !existing_files.contains(&(file.file_name.to_string(), file.file_path.to_string())) {
                    match insert_stmt.execute(params![
                        file.file_name,
                        file.file_path,
                        file.file_type.as_deref().map::<&str, _>(|s| s.as_ref())
                    ]) {
                        Ok(_) => (),
                        Err(e) => return Err(e.to_string())
                    };
                    inserted_count += 1;
                }
            }
            if (i + 1) % 10 == 0 {
                // Progress reporting can be added here if needed
            }
        }
        println!("File insertion completed in {:?}", insert_start.elapsed());
        println!("Total files inserted: {}", inserted_count);
    }
    println!("Committing transaction");
    let commit_start = Instant::now();
    match tx.commit() {
        Ok(_) => {},
        Err(e) => return Err(e.to_string())
    };
    println!("Transaction committed in {:?}", commit_start.elapsed());

    println!("create_database function completed in {:?}", start_time.elapsed());
    Ok(())
}

pub fn check_database(
    mut conn: PooledConnection<SqliteConnectionManager>,
    allowed_file_extensions: &HashSet<String>,
    pool: &rayon::ThreadPool,
) -> Result<(), Box<dyn std::error::Error>> {
    let bad_paths: Vec<String> = {
        let mut stmt = conn.prepare_cached("SELECT file_path FROM files")?;
        let rows: Vec<Result<String, _>> = stmt.query_map([], |row| row.get::<_, String>(0))?.collect();

        pool.install(|| {
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
    conn: &PooledConnection<SqliteConnectionManager>,
    search_term: &str,
    similarity_threshold: f64,
    thread_pool: &rayon::ThreadPool,
    searchpath: PathBuf
) -> Result<Vec<DirEntry>> {
    let start_time = Instant::now();

    // Convert searchpath to a string
    let search_path_str = searchpath.to_str().unwrap_or("");

    // Modify the SQL query to filter by path
    let mut stmt = conn.prepare("SELECT file_name, file_path, file_type FROM files WHERE file_path LIKE ?")?;
    let rows = stmt.query_map(&[&format!("{}%", search_path_str)], |row| Ok((
        row.get::<_, String>(0)?,
        row.get::<_, String>(1)?,
        row.get::<_, Option<String>>(2)?
    )))?;

    let (tx, rx) = channel();
    let file_data: Vec<(String, String, Option<String>)> = rows.collect::<Result<Vec<_>>>()?;
    thread_pool.install(|| {
        file_data.into_par_iter().for_each(|(file_name, file_path, file_type)| {
            let tx = tx.clone();
            let search_term = search_term.to_owned();
            let name_to_compare = if file_type.as_deref() == Some("directory") {
                &file_name
            } else {
                &file_name.split('.').next().unwrap_or(&file_name).to_string()
            };
            let similarity = normalized_levenshtein(&name_to_compare, &search_term);
            if similarity >= similarity_threshold {
                tx.send((file_path, similarity)).unwrap();
            }
        });
    });
    drop(tx);

    let mut results: Vec<(String, f64)> = rx.iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let return_entries: Vec<DirEntry> = results.into_iter()
        .filter_map(|(s, _)| {
            let path = Path::new(&s);
            fs::read_dir(path.parent().unwrap_or(Path::new(".")))
                .ok()
                .and_then(|mut dir| dir.find(|e| e.as_ref().map(|d| d.path() == path).unwrap_or(false)))
                .and_then(|e| e.ok())
        })
        .collect();

    let duration = start_time.elapsed();
    println!("Parallel search completed in {:.2?}", duration);

    Ok(return_entries)
}


fn convert_to_forward_slashes(path: &PathBuf) -> String {
    path.to_str()
        .map(|s| s.replace('\\', "/"))
        .unwrap_or_else(|| String::new())
}
