use crate::config_handler::{
    get_embedding_dimensions, get_number_results_embedding, get_number_results_levenshtein,
    get_path_to_vocab, get_path_to_weights, CURRENT_DIR,
};
use crate::db_create::create_database;
use crate::db_search::search_database;
use crate::db_util::{initialize_database, load_vocab};
use crate::file_information::{get_file_information, FileData, FileDataFormatted};
use bytemuck::cast_slice;
use log::{error, info};
use ndarray::Array2;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use std::{
    collections::HashMap,
    fs::{self, DirEntry},
    path::PathBuf,
    sync::OnceLock,
};
use tauri::{command, AppHandle, State};

#[derive(Debug)]
pub struct AppState {
    pub handle: AppHandle,
}

pub static WEIGHTS: OnceLock<Array2<f32>> = OnceLock::new();
pub static VOCAB: OnceLock<HashMap<String, usize>> = OnceLock::new();

#[derive(Debug)]
struct SqliteCustomizer;

impl r2d2::CustomizeConnection<rusqlite::Connection, rusqlite::Error> for SqliteCustomizer {
    fn on_acquire(&self, conn: &mut rusqlite::Connection) -> Result<(), rusqlite::Error> {
        conn.execute_batch(
            "PRAGMA mmap_size = 268435456;
             PRAGMA cache_size = -64000;
             PRAGMA synchronous = OFF;
             PRAGMA journal_mode = WAL;",
        )
    }
}

/// Initializes VOCAB and WEIGHTS to be their respective files
pub fn initialize_globals() {
    info!("Initializing globals");
    WEIGHTS.get_or_init(|| {
        let embedding_dim = get_embedding_dimensions(); // Currently returns 300 (WRONG)
        let weights_path = get_path_to_weights();
        let filename = weights_path.to_string_lossy();

        let weights_bytes = fs::read(&weights_path).unwrap_or_else(|e| {
            error!("Could not read weights: {e}");
            Vec::new()
        });

        if weights_bytes.is_empty() { return Array2::zeros((0, 0)); }

        let weights_f32: Vec<f32> = if filename.contains("_Q8") {
            let scale = crate::config_handler::get_q8_scale();
            weights_bytes.iter().map(|&b| (b as i8 as f32) * (scale / 127.0)).collect()
        } else if filename.contains("_Q16") {
            cast_slice::<u8, half::f16>(&weights_bytes).iter().map(|f| f.to_f32()).collect()
        } else {
            cast_slice::<u8, f32>(&weights_bytes).to_vec()
        };

        let total_elements = weights_f32.len();

        // Check if the data actually fits the dimensions
        if total_elements % embedding_dim != 0 {
            error!(
                "CRITICAL: Model file has {} elements, which is not divisible by dim {}.
                Is your config.json set to 150?",
                total_elements, embedding_dim
            );
            return Array2::zeros((0, 0));
        }

        let vocab_size = total_elements / embedding_dim;
        info!("Model Loaded: Vocab={}, Dim={}", vocab_size, embedding_dim);

        Array2::from_shape_vec((vocab_size, embedding_dim), weights_f32)
            .unwrap_or_else(|e| {
                error!("Shape error: {e}");
                Array2::zeros((0, 0))
            })
    });

    VOCAB.get_or_init(|| load_vocab(&get_path_to_vocab()));
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

    if !path.exists() {
        let _ = fs::create_dir_all(&path);
    }

    path.push("files.sqlite3");
    let manager = SqliteConnectionManager::file(path);

    Pool::builder()
        .connection_customizer(Box::new(SqliteCustomizer))
        .build(manager)
        .expect("Failed to create pool")
}

/// Populates the database with the files which are under the Path given
pub fn manager_populate_database(database_scan_start: PathBuf) -> Result<(), String> {
    let connection_pool = manager_make_connection_pool();

    if let Ok(conn) = connection_pool.get() {
        initialize_database(&conn);
    } else {
        error!("Initializing failed: could not get connection from pool");
        return Err("Initializing failed: could not get connection from pool".to_string());
    }

    let pooled_connection = match connection_pool.get() {
        Ok(conn) => conn,
        Err(e) => {
            error!("Failed to get pooled connection: {e}");
            return Err(e.to_string());
        }
    };

    if let Err(e) = pooled_connection.pragma_update(None, "journal_mode", "WAL") {
        error!("journal_mode failed: {e}");
    }
    if let Err(e) = pooled_connection.pragma_update(None, "synchronous", "NORMAL") {
        error!("synchronous failed: {e}");
    }
    if let Err(e) = pooled_connection.pragma_update(None, "wal_autocheckpoint", "1000") {
        error!("wal auto checkpoint failed: {e}");
    }

    match create_database(connection_pool, database_scan_start) {
        Ok(_) => {}
        Err(e) => return Err(e.to_string()),
    };

    Ok(())
}

/// starts the search with a search term, location, extensions and sends it to FrontEnd via an Event
/// search filetype is the Filetype Ending without the Dot; for Directory's it must be a dir
#[command(async)]
pub fn manager_basic_search(
    searchterm: &str,
    searchpath: &str,
    searchfiletype: String,
    state: State<AppState>,
) {
    initialize_globals();
    info!("search started !");
    let connection_pool = manager_make_connection_pool();

    let search_path = PathBuf::from(searchpath);

    search_database(
        connection_pool,
        searchterm,
        search_path,
        searchfiletype,
        get_number_results_embedding(),
        get_number_results_levenshtein(),
        state,
    );
}

pub fn check_for_default_paths() {
    info!("checking for default paths");

    // Model weights check
    let model_path = CURRENT_DIR.clone().join("data/model/eng_weights_D300");
    if !model_path.exists() {
        error!(
            "The default weights file couldn't be found at {:?}",
            CURRENT_DIR.clone().join("data/model/eng_weights_D300")
        );
    }

    // Vocab check
    let vocab_path = CURRENT_DIR.clone().join("data/model/eng_vocab.json");
    if !vocab_path.exists() {
        error!(
            "The default vocab file couldn't be found at {:?}",
            CURRENT_DIR.clone().join("data/model/eng_vocab.json")
        );
    }
}
