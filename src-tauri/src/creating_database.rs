use std::sync::mpsc::channel;
use rusqlite::{params, Result};
use threadpool::ThreadPool;
use walkdir::{WalkDir};
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;
use std::sync::Arc;
use std::sync::Mutex;


#[derive(Debug)]
struct Files {
    id: i32,
    file_name: String,
    file_path: String,
    file_type: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let manager = SqliteConnectionManager::file("files.sqlite3");
    let pool = Pool::new(manager)?;
    let conn = pool.get()?;
    let thread_count = 8;

    let _result = conn.execute(
        "CREATE TABLE IF NOT EXISTS files (
        id   INTEGER PRIMARY KEY,
        file_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_type  BLOB
    )",
        ())?;

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

    let create_db_start = Instant::now();
    let _ = create_database(conn, "/", thread_count, &allowed_file_extensions)?;
    println!("Database creation took: {:?}", create_db_start.elapsed());

    let conn = pool.get()?;

    let check_db_start = Instant::now();
    checking_database(conn, thread_count, &allowed_file_extensions)?;
    println!("Database checking took: {:?}", check_db_start.elapsed());

    println!("Total execution time: {:?}", start_time.elapsed());
    Ok(())
}


fn create_database(
    conn: PooledConnection<SqliteConnectionManager>,
    path: &str,
    n_workers: usize,
    allowed_file_extensions: &HashSet<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut files_vec: Vec<Files> = Vec::new();
    let pool = ThreadPool::new(n_workers);
    let (tx, rx) = channel();
    let mut conn = conn;

    for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
        let tx = tx.clone();
        pool.execute(move || {
            tx.send(entry).unwrap();
        });
    }

    drop(tx);
    for entry in rx {
        let path = entry.path();
        if !should_ignore_path(path) && (path.is_dir() || is_allowed_file(path, allowed_file_extensions)) {
            let file_name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            let file_path = path.to_string_lossy().to_string();
            let file_type = if path.is_dir() {
                Some("directory".to_string())
            } else {
                path.extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
            };

            let me = Files {
                id: 0,
                file_name,
                file_path,
                file_type,
            };
            files_vec.push(me);
        }
    }


    println!("Number of files to insert: {}", files_vec.len());

    let tx = conn.transaction()?;
    {
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

        let mut insert_stmt = tx.prepare_cached("INSERT INTO files (file_name, file_path, file_type) VALUES (?, ?, ?)")?;
        let batch_size = 1000;
        for chunk in files_vec.chunks(batch_size) {
            for file in chunk {
                if !existing_files.contains(&(file.file_name.clone(), file.file_path.clone())) {
                    insert_stmt.execute(params![
                        file.file_name,
                        file.file_path,
                        file.file_type.as_deref()
                    ])?;
                }
            }
        }
    }
    tx.commit()?;
    Ok(())
}


fn get_column_as_vec(conn: &PooledConnection<SqliteConnectionManager>, column_name: &str, table_name: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(&format!("SELECT {} FROM {}", column_name, table_name))?;

    let column_data = stmt.query_map([], |row| row.get(0))?
        .collect::<Result<Vec<String>>>()?;

    Ok(column_data)
}

fn checking_database(
    conn: PooledConnection<SqliteConnectionManager>,
    n_workers: usize,
    allowed_file_extensions: &HashSet<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let conn = Arc::new(Mutex::new(conn));
    let pool = ThreadPool::new(n_workers);
    let (sender, receiver) = channel();
    let allowed_file_extensions = Arc::new(allowed_file_extensions.clone());

    {
        let conn = conn.lock().unwrap();
        let mut stmt = conn.prepare_cached("SELECT file_path FROM files")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;

        for path in rows {
            let path = path?;
            let sender = sender.clone();
            let allowed_extensions = Arc::clone(&allowed_file_extensions);
            pool.execute(move || {
                let path_obj = Path::new(&path);
                if !is_allowed_file(path_obj, &allowed_extensions) &&
                    !std::fs::metadata(&path).is_ok() {
                    sender.send(path).unwrap();
                }
            });
        }
    }

    drop(sender);

    let bad_paths: Vec<String> = receiver.iter().collect();
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