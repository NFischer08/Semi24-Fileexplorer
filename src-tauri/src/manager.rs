use crate::config_handler::{get_embedding_dimensions, get_number_results_embedding, get_number_results_levenhstein, get_path_to_vocab, get_path_to_weights, CURRENT_DIR};
use crate::db_create::create_database;
use crate::db_search::search_database;
use crate::db_util::{initialize_database, load_vocab};
use crate::file_information::{get_file_information, FileData, FileDataFormatted};
use bytemuck::cast_slice;
use ndarray::Array2;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::HashMap;
use std::fs;
use std::fs::create_dir;
use std::sync::{LazyLock, OnceLock};
use std::{fs::DirEntry, path::PathBuf};
use tauri::command;
use tauri::{AppHandle, State};

#[derive(Debug)]
pub struct AppState {
    pub handle: AppHandle,
}

pub static WEIGHTS: OnceLock<Array2<f32>> = OnceLock::new();
pub static VOCAB: OnceLock<HashMap<String, usize>> = OnceLock::new();

pub static THREAD_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap()
});

/// Initializes VOCAB and WEIGHTS to be their respective files
pub fn initialize_globals() {
    WEIGHTS.get_or_init(|| {
        let wights_pth = CURRENT_DIR.clone().join("data/model/eng_weights_D300");

        println!("{:?}", wights_pth);
        
        let embedding_dim = get_embedding_dimensions();

        let weights_bytes: Vec<u8> = fs::read(get_path_to_weights()).expect("Could not read weights");
        let weights_as_f32: &[f32] = cast_slice(&weights_bytes);

        // Infer the vocab size from the file length
        let vocab_size = weights_as_f32.len() / embedding_dim;

        Array2::from_shape_vec((vocab_size, embedding_dim), weights_as_f32.to_vec())
            .expect("Shape mismatch in weights")
    });

    VOCAB.get_or_init(|| {
        load_vocab(&get_path_to_vocab())
    });
}

/// Builds up the FileDataFormatted Struct from DireEntries
pub fn build_struct(entries: Vec<DirEntry>) -> Vec<FileDataFormatted> {
    entries
        .into_iter()
        .map(|entry| FileData::format(get_file_information(&entry)))
        .collect()
}

/// Creates the connection pool to the Database which is called files.sqlite3
pub fn manager_make_connection_pool() -> Pool<SqliteConnectionManager> {
    let mut path = CURRENT_DIR.clone();
    path.push("data/db");
    if PathBuf::from(&path).try_exists().expect("Reason") {
        path.push("files.sqlite3");
        let manager = SqliteConnectionManager::file(path);
        Pool::new(manager).expect("Failed to create pool.")
        
    } else {
        create_dir(PathBuf::from(&path)).expect("Failed to create Dir");
        path.push("files.sqlite3");
        let manager = SqliteConnectionManager::file(path);
        Pool::new(manager).expect("Failed to create pool.")
    }
}

/// Populates the database with the files which are under the Path given
pub fn manager_populate_database(database_scan_start: PathBuf) -> Result<(), String> {
    initialize_globals();
    let connection_pool = manager_make_connection_pool();

    initialize_database(&connection_pool.get().expect("Initializing failed: "));

    let pooled_connection = connection_pool.get().unwrap();

    pooled_connection
        .pragma_update(None, "journal_mode", "WAL")
        .expect("journal_mode failed");
    pooled_connection
        .pragma_update(None, "synchronous", "NORMAL")
        .expect("synchronous failed");
    pooled_connection
        .pragma_update(None, "wal_autocheckpoint", "1000")
        .expect("wal_autocheckpoint failed");

    match create_database(connection_pool, database_scan_start) {
        Ok(_) => {}
        Err(e) => return Err(e.to_string()),
    };

    Ok(())
}

/// starts the search with a search term, location, extensions and sends it to FrontEnd via an Event
/// searchfiletype is the Filetype Ending without the Dot, for Directorys it must be dir
#[command(async)]
pub fn manager_basic_search(
    searchterm: &str,
    searchpath: &str,
    searchfiletype: &str,
    state: State<AppState>,
) {
    initialize_globals();
    println!("search started !");
    let connection_pool = manager_make_connection_pool();

    let search_path = PathBuf::from(searchpath);

    search_database(
        connection_pool,
        searchterm,
        search_path,
        searchfiletype,
        get_number_results_embedding(),
        get_number_results_levenhstein(),
        state,
    );
}
