mod database_operations;

use rayon::ThreadPoolBuilder;
use std::path::PathBuf;
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use database_operations::{initialize_database_and_extensions, create_database, check_database, search_database };

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
) -> Result<(Vec<String>), Box<dyn std::error::Error>>
{
    let pooled_connection= manager_make_pooled_connection()?;

    let similarity_threshold = 0.7;
    let threads = num_cpus::get();

    let return_paths = search_database(&pooled_connection, search_term, similarity_threshold, threads)?;  // Hier kann das Frontend abgreifen
    Ok(return_paths)
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