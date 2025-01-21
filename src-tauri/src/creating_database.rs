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

    println!("Creating database");
    let manager = SqliteConnectionManager::file("files.sqlite3");
    let pool = Pool::new(manager)?;
    let conn = pool.get()?;

    let pragma_check: String = conn.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;
    if pragma_check != "ok" {
        println!("Database integrity check failed: {}", pragma_check);
    }

    println!("Creating tables");
    let result = conn.execute(
        "CREATE TABLE IF NOT EXISTS files (
        id   INTEGER PRIMARY KEY,
        file_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_type  BLOB
    )",
        ())?;

    println!("SQL Message Creating Table: {}", result);

    let _ = create_database(conn, "/", 8)?;


    let conn = pool.get()?;

    //checking_database(conn);

    Ok(())
}

fn create_database(
    conn: PooledConnection<SqliteConnectionManager>,
    path: &str, // Path to starting Folder of Indexing
    n_workers: usize,
) -> Result<(), Box<dyn std::error::Error>> {

    println!("Populating Database");
    let mut files_vec: Vec<Files> = Vec::new();
    let pool = ThreadPool::new(n_workers);
    let (tx, rx) = channel(); // Channel for communication between threads
    let mut conn = conn;

    println!("Iterating Entries");
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


        let me = Files {
            id: 0,
            file_name,
            file_path,
            file_type,
        };
        files_vec.push(me);
    }
    println!("Number of files to insert: {}", files_vec.len());

    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare_cached("INSERT INTO files (file_name, file_path, file_type) VALUES (?, ?, ?)")?;
        for (index, file) in files_vec.iter().enumerate() {
            match stmt.execute(params![
            &file.file_name,
            &file.file_path,
            &file.file_type,
        ]) {
                Ok(_) => println!("Inserted file {}", index),
                Err(e) => eprintln!("Error inserting file {}: {:?}", index, e),
            }
        }
    }
    tx.commit()?;

    println!("Done with population");

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
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Checking database");
    let file_paths = get_column_as_vec(&conn, "file_path", "files");
    let vec = file_paths.unwrap();
    for path in vec {
        println!("{}",std::fs::exists(&path).unwrap());
        if (std::fs::exists(&path).unwrap() == false) {
            //let rows_deleted = conn.execute(
            //    "DELETE FROM files WHERE file_path = ?1",
            //    [path.as_str()],
            //)?;
            //println!("rows_deleted: {}", rows_deleted);
            println!("{}", path);
        }
    }
    Ok(())
}