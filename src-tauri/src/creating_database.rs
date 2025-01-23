use std::sync::mpsc::channel;
use rusqlite::{params, Result};
use threadpool::ThreadPool;
use walkdir::{WalkDir};
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug)]
struct Files {
    id: i32,
    file_name: String,
    file_path: String,
    file_type: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        "txt", "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
        "jpg", "jpeg", "png", "gif", "mp3", "mp4", "avi", "mov",
        "zip", "rar", "7z", "tar", "gz", "csv", "json", "xml",
        "html", "htm", "css", "js", "py", "java", "c", "cpp", "h",
        "rs", "go", "php", "rb", "pl", "sh", "bat", "ps1"
    ].iter().map(|&s| s.to_string()).collect();

    let _ = create_database(conn, "/", thread_count, &allowed_file_extensions)?;

    let conn = pool.get()?;

    checking_database(conn, thread_count, &allowed_file_extensions)?;
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
        if is_allowed_file(path, allowed_file_extensions) && !should_ignore_path(path) {
            let file_name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            let file_path = path.to_string_lossy().to_string();
            let file_type = path.extension()
                .and_then(|s| s.to_str())
                .unwrap_or("").to_string();

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
        let mut existing_files = std::collections::HashSet::new();
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
                file.file_type
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
    let mut conn = conn;

    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare_cached("SELECT file_path FROM files")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;

        let mut bad_paths = Vec::new();

        let pool = ThreadPool::new(n_workers);
        let (sender, receiver) = channel();

        for path in rows {
            let path = path?;
            let sender = sender.clone();
            let allowed_file_extensions = allowed_file_extensions.clone();
            pool.execute(move || {
                let path_obj = Path::new(&path);
                if !is_allowed_file(path_obj, &allowed_file_extensions) &&
                    !std::fs::metadata(&path).is_ok() {
                    sender.send(path).unwrap();
                }
            });
        }

        drop(sender); // Close the sender

        for path in receiver {
            bad_paths.push(path);
        }

        println!("bad files: {:?}", bad_paths);
        println!("Number of bad files: {}", bad_paths.len());

        // Optionally, remove bad paths from the database
        //let mut delete_stmt = tx.prepare_cached("DELETE FROM files WHERE file_path = ?")?;
        //for path in bad_paths {
        //    delete_stmt.execute([&path])?;
        //}
    }
    tx.commit()?;

    Ok(())
}

fn is_allowed_file(path: &Path, allowed_file_extensions: &HashSet<String>) -> bool {
    if should_ignore_path(path) {
        return false;
    }
    if path.is_dir() {
        return true;
    }
    path.extension()
        .and_then(|s| s.to_str())
        .map(|ext| allowed_file_extensions.contains(ext))
        .unwrap_or(false)
}


fn should_ignore_path(path: &Path) -> bool {
    path.to_str().map_or(false, |s| s.starts_with("/proc") || s.starts_with("/sys"))
}
