use std::sync::mpsc::channel;
use rusqlite::{params, Result};
use threadpool::ThreadPool;
use walkdir::{WalkDir};
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;

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

    //println!("SQL Message Creating Table: {}", _result);

    let system_file_extensions = vec![
        ".o".to_string(), // Objektdatei mit kompiliertem Code
        ".so".to_string(), // Gemeinsam genutzte Bibliotheksdatei unter Unix-Systemen
        ".pem".to_string(), // Textdatei für Zertifikate und kryptografische Schlüssel
        ".lock".to_string(), // Sperrdatei zur Verhinderung gleichzeitiger Zugriffe3
        ".so.1".to_string(), // Versionierte gemeinsam genutzte Bibliotheksdatei5
        "".to_string(), // Keine Dateiendung
    ];

    let _ = create_database(conn, "/", thread_count, &system_file_extensions)?;


    let conn = pool.get()?;

    checking_database(conn, thread_count, &system_file_extensions)?;
    Ok(())
}

fn create_database(
    conn: PooledConnection<SqliteConnectionManager>,
    path: &str, // Path to starting Folder of Indexing
    n_workers: usize,
    system_file_extensions: &Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {

    let mut files_vec: Vec<Files> = Vec::new();
    let pool = ThreadPool::new(n_workers);
    let (tx, rx) = channel(); // Channel for communication between threads
    let mut conn = conn;

    for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) { //This Line does not Throw an Error if the Path supplied is not valid careful
        let tx= tx.clone();

        pool.execute(move || {
            let full_path = entry.path().to_owned();
            tx.send(full_path).unwrap();
        });
    }
    drop(tx);

    for received in rx {
        let file_name = received.file_name().unwrap_or_default().to_string_lossy().to_string();
        let file_type = received.extension().unwrap_or_default().to_string_lossy().to_string();
        let file_path = received.as_path().to_path_buf().to_str().unwrap().to_string();

        // Überprüfen, ob die Dateiendung in der Liste der Systemdateien ist
        if !system_file_extensions.contains(&file_type) {
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

        // Insert files that don't exist
        for (index, file) in files_vec.iter().enumerate() {
            if !existing_files.contains(&(file.file_name.clone(), file.file_path.clone())) {
                match insert_stmt.execute(params![
                &file.file_name,
                &file.file_path,
                &file.file_type,
            ]) {
                    Ok(_) => {},
                    Err(e) => eprintln!("Error inserting file {}: {:?}", index, e),
                }
            } else {
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
    system_file_extensions: &Vec<String>,
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
            let system_file_extensions = system_file_extensions.clone();
            pool.execute(move || {
                if !is_system_file(&path, &system_file_extensions) &&
                    !std::fs::metadata(&path).is_ok() {
                    sender.send(path).unwrap();
                }
            });
        }

        drop(sender); // Close the sender

        for path in receiver {
            bad_paths.push(path);
        }

        //println!("bad files: {:?}", bad_paths);
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

fn is_system_file(path: &str, system_file_extensions: &[String]) -> bool {
    system_file_extensions.iter().any(|ext| path.ends_with(ext))
}