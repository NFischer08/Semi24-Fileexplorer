use r2d2::Pool;
use rusqlite::{Connection, Result};
use r2d2_sqlite::SqliteConnectionManager;
use std::sync::mpsc::channel;
use threadpool::ThreadPool;
use strsim::normalized_levenshtein;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manager = SqliteConnectionManager::file("files.sqlite3");
    let pool = Pool::new(manager)?;
    let conn = pool.get()?;

    let search_term = "Test";
    let similarity_threshold = 0.7;
    let n_workers = 8;

    match find_similar_matches_parallel(&conn, search_term, similarity_threshold, n_workers) {
        Ok(file_paths) => {
            if file_paths.is_empty() {
                println!("No similar files found for: {}", search_term);
            } else {
                println!("Similar files found for '{}':", search_term);
                for path in file_paths {
                    println!("  {}", path);
                }
            }
        },
        Err(e) => println!("Error searching for similar files: {}", e),
    }

    Ok(())
}

fn find_similar_matches_parallel(
    conn: &Connection,
    search_term: &str,
    similarity_threshold: f64,
    n_workers: usize,
) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT file_name, file_path FROM files")?;
    let rows = stmt.query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?;

    let (tx, rx) = channel();
    let pool = ThreadPool::new(n_workers);

    for row in rows {
        if let Ok((file_name, file_path)) = row {
            let tx = tx.clone();
            let search_term = search_term.to_owned();

            pool.execute(move || {
                let similarity = normalized_levenshtein(&file_name, &search_term);
                if similarity >= similarity_threshold {
                    tx.send(file_path).unwrap();
                }
            });
        }
    }
    drop(tx);

    let results: Vec<String> = rx.iter().collect();
    Ok(results)
}
