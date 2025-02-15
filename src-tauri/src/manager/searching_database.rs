use rusqlite::{Result};
use r2d2::{PooledConnection};
use std::sync::mpsc::channel;
use rayon::prelude::*;
use strsim::normalized_levenshtein;
use std::time::Instant;
use r2d2_sqlite::SqliteConnectionManager;

pub fn search_database(
    conn: &PooledConnection<SqliteConnectionManager>,
    search_term: &str,
    similarity_threshold: f64,
    thread_pool: &rayon::ThreadPool,
) -> Result<Vec<String>> {

    let start_time = Instant::now();
    let mut stmt = conn.prepare("SELECT file_name, file_path, file_type FROM files")?;
    let rows = stmt.query_map([], |row| Ok((
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
                file_name
            } else {
                file_name.split('.').next().unwrap_or(&file_name).to_string()
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
    let return_paths: Vec<String> = results.iter().map(|p| p.0.clone()).collect();
    let duration = start_time.elapsed();
    println!("Parallel search completed in {:.2?}", duration);

    Ok(return_paths)
}
