mod database_operations;
pub mod file_information;

use std::fs::DirEntry;
use rayon::ThreadPoolBuilder;
use std::path::PathBuf;
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use database_operations::{check_database, create_database, initialize_database_and_extensions, search_database};
use file_information::{get_file_information, };

#[derive(Debug, serde::Serialize)] // TODO: braucht man diese Zeile?
pub struct SearchResult {
    name: String,
    path: String,
    last_modified: String,
    file_type: String,
    size: String
}
fn build_struct(paths: Vec<DirEntry>) -> Vec<SearchResult> {
    paths.into_iter()
        .map(|path| {
            let file_entry = get_file_information(&path);
            SearchResult {
                name: file_entry.name,
                path: path.path().to_string_lossy().into_owned(),
                last_modified: file_entry.last_modified.to_rfc3339(),
                file_type: format!("{:?}", file_entry.file_type),
                size: format!("{} KB", file_entry.size_in_kb)
            }
        })
        .collect()
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // This is just for testing

    let database_scan_start = "/";
    manager_create_database(database_scan_start)?;
    manager_basic_search("Test")?;
    manager_check_database()?;
    Ok(())
}
fn manager_make_pooled_connection() -> Result<PooledConnection<SqliteConnectionManager>, Box<dyn std::error::Error>> {
    let manager = SqliteConnectionManager::file("files.sqlite3");
    let connection_pool = Pool::new(manager)?;
    let pooled_connection = connection_pool.get()?;

    Ok(pooled_connection)
}
pub fn manager_create_database(
    database_scan_start: &str,
)  -> Result<(), Box<dyn std::error::Error>>
{
    let pooled_connection= manager_make_pooled_connection()?;

    let allowed_file_extensions= initialize_database_and_extensions(&pooled_connection)?;

    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap();

    let db_path = PathBuf::from(database_scan_start);

    create_database(pooled_connection, db_path, &allowed_file_extensions, &thread_pool)?;

    Ok(())
}

pub fn manager_basic_search(
    search_term: &str,
) -> Result<(Vec<SearchResult>), Box<dyn std::error::Error>>
{
    let pooled_connection= manager_make_pooled_connection()?;

    let similarity_threshold = 0.7;

    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap();

    let return_paths = search_database(&pooled_connection, search_term, similarity_threshold, &thread_pool)?;  // Hier kann das Frontend abgreifen
    let search_result = build_struct(return_paths);
    Ok(search_result)
}

pub fn manager_check_database() -> Result<(), Box<dyn std::error::Error>> {
    let pooled_connection= manager_make_pooled_connection()?;

    let allowed_file_extensions= initialize_database_and_extensions(&pooled_connection)?;
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap();

    check_database(pooled_connection, &allowed_file_extensions, &thread_pool)?;

    Ok(())
}