use std::sync::{Arc, Mutex};
use rusqlite::{params, Result};
use rayon::prelude::*;
use jwalk::WalkDir;
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;


#[derive(Debug)]
struct Files {
    id: i32,
    file_name: String,
    file_path: String,
    file_type: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Main is running");
    let start_time = Instant::now();

    let manager = SqliteConnectionManager::file("files.sqlite3");
    let pool = Pool::new(manager)?;
    let conn = pool.get()?;
    println!("Main is still running");


    let _result = conn.execute(
        "CREATE TABLE IF NOT EXISTS files (
        id   INTEGER PRIMARY KEY,
        file_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_type  BLOB
    )",
        ())?;

    println!("Main is still running");
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON files (file_path)", [])?;

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
        "sql", "db", "sqlite", "mdb", "ttf", "otf", "woff", "woff2", "obj", "stl", "fbx", "dxf", "dwg", "psd", "ai", "indd", "iso", "img", "dmg", "bak", "tmp", "log", "pcap"
    ].iter().map(|&s| String::from(s)).collect();

    println!("Main is still running");
    let create_db_start = Instant::now();
    let _ = create_database(conn, r"C:\Users\maxmu", &allowed_file_extensions)?;
    println!("Database creation took: {:?}", create_db_start.elapsed());

    let conn = pool.get()?;

    let check_db_start = Instant::now();
    checking_database(conn, &allowed_file_extensions)?;
    println!("Database checking took: {:?}", check_db_start.elapsed());

    println!("Total execution time: {:?}", start_time.elapsed());
    Ok(())
}


fn create_database(
    conn: PooledConnection<SqliteConnectionManager>,
    path: &str,
    allowed_file_extensions: &HashSet<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting create_database function");
    println!("Scanning directory: {}", path);

    let start_time = std::time::Instant::now();
    let files_vec: Vec<Files> = WalkDir::new(path)
        .parallelism(jwalk::Parallelism::RayonNewPool(num_cpus::get()))
        .into_iter()
        .filter_map(|entry_result| {
            entry_result.ok().and_then(|entry| {
                let path = entry.path();
                if !should_ignore_path(&path) && (path.is_dir() || is_allowed_file(&path, allowed_file_extensions)) {
                    Some(Files {
                        id: 0,
                        file_name: entry.file_name().to_string_lossy().to_string(),
                        file_path: path.to_string_lossy().to_string(),
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
        .collect();

    println!("Directory scan completed in {:?}", start_time.elapsed());
    println!("Number of files to insert: {}", files_vec.len());

    let mut conn = conn;
    let tx = conn.transaction()?;
    {
        println!("Starting to fetch existing files");
        let fetch_start = std::time::Instant::now();
        let mut existing_files = HashSet::new();
        let mut stmt = tx.prepare_cached("SELECT file_name, file_path FROM files")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;

        for row in rows {
            if let Ok((name, path)) = row {
                existing_files.insert((name, path));
            }
        }
        println!("Fetched {} existing files in {:?}", existing_files.len(), fetch_start.elapsed());

        println!("Starting file insertion");
        let insert_start = std::time::Instant::now();
        let mut insert_stmt = tx.prepare_cached("INSERT INTO files (file_name, file_path, file_type) VALUES (?, ?, ?)")?;
        let batch_size = 1000;
        let mut inserted_count = 0;
        for (i, chunk) in files_vec.chunks(batch_size).enumerate() {
            for file in chunk {
                if !existing_files.contains(&(file.file_name.clone(), file.file_path.clone())) {
                    insert_stmt.execute(params![
                        file.file_name,
                        file.file_path,
                        file.file_type.as_deref()
                    ])?;
                    inserted_count += 1;
                }
            }
            if (i + 1) % 10 == 0 {
            }
        }
        println!("File insertion completed in {:?}", insert_start.elapsed());
        println!("Total files inserted: {}", inserted_count);
    }
    println!("Committing transaction");
    let commit_start = std::time::Instant::now();
    tx.commit()?;
    println!("Transaction committed in {:?}", commit_start.elapsed());

    println!("create_database function completed in {:?}", start_time.elapsed());
    Ok(())
}


fn checking_database(
    conn: PooledConnection<SqliteConnectionManager>,
    allowed_file_extensions: &HashSet<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let conn = Arc::new(Mutex::new(conn));
    let allowed_file_extensions = Arc::new(allowed_file_extensions.clone());

    let bad_paths: Vec<String> = {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare_cached("SELECT file_path FROM files")?;
        let rows: Vec<Result<String, _>> = stmt.query_map([], |row| row.get::<_, String>(0))?.collect();

        rows.into_par_iter()
            .filter_map(|path_result| {
                let path = path_result.ok()?;
                let path_obj = Path::new(&path);
                if !is_allowed_file(path_obj, &allowed_file_extensions) &&
                    !std::fs::metadata(&path).is_ok() {
                    Some(path)
                } else {
                    None
                }
            })
            .collect()
    };

    println!("Number of bad files: {}", bad_paths.len());

    if !bad_paths.is_empty() {
        let mut conn = conn.lock().unwrap();
        let tx = conn.transaction()?;
        {
            let placeholders = bad_paths.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let query = format!("DELETE FROM files WHERE file_path IN ({})", placeholders);
            let mut stmt = tx.prepare_cached(&query)?;
            stmt.execute(rusqlite::params_from_iter(bad_paths.iter()))?;
        }
        tx.commit()?;
    }

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