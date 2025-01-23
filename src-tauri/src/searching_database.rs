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
                println!("Similar files found for '{}' (sorted by similarity):", search_term);
                for (path, file_type, full_file_name, similarity) in file_paths {
                    println!("  {} (full name: {}, type: {}, similarity: {:.4})", path, full_file_name, file_type, similarity);
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
) -> Result<Vec<(String, String, String, f64)>> {
    let mut stmt = conn.prepare("SELECT file_name, file_path, file_type FROM files")?;

    let rows = stmt.query_map([], |row| Ok((
        row.get::<_, String>(0)?,
        row.get::<_, String>(1)?,
        row.get::<_, Option<String>>(2)?
    )))?;

    let (tx, rx) = channel();
    let pool = ThreadPool::new(n_workers);

    for row in rows {
        if let Ok((file_name, file_path, file_type)) = row {
            let tx = tx.clone();
            let search_term = search_term.to_owned();

            pool.execute(move || {
                let name_to_compare = if file_type.as_deref() == Some("directory") {
                    file_name.clone()
                } else {
                    file_name.split('.').next().unwrap_or(&file_name).to_string()
                };
                let similarity = normalized_levenshtein(&name_to_compare, &search_term);
                if similarity >= similarity_threshold {
                    tx.send((file_path, file_type.unwrap_or_else(|| "Unknown".to_string()), file_name, similarity)).unwrap();
                }
            });
        }
    }
    drop(tx);

    let mut results: Vec<(String, String, String, f64)> = rx.iter().collect();
    results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

    Ok(results)
}
