mod creating_database;
mod searching_database;

use rayon::ThreadPoolBuilder;
use std::path::PathBuf;
use creating_database::{ initialize_database_and_extensions, create_database, check_database };
use searching_database:: {search_database};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the database
    let (connection_pool, allowed_file_extensions) = initialize_database_and_extensions()?;

    // Create a thread pool
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap();

    // Set the path to scan
    let db_path = PathBuf::from("/");

    // Get a connection from the pool
    let conn = connection_pool.get()?;

    // Create the database
    create_database(&conn, db_path, &allowed_file_extensions, &thread_pool)?;

    // Get another connection from the pool
    let conn = connection_pool.get()?;

    // Check the database
    check_database(&conn, &allowed_file_extensions, &thread_pool)?;

    let found_paths = search_database(&conn, "Test", 0.8, num_cpus::get())?;

    println!("{:?}", found_paths);

    Ok(())
}
