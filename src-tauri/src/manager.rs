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
    fs::{self, create_dir, DirEntry},
    path::PathBuf,
    sync::OnceLock,
};
use tauri::{command, AppHandle, State};

/// Simple memory usage logging function
pub fn log_memory_usage(context: &str) {
    // Read from /proc/self/status on Linux
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                if let Some(memory_str) = line.split_whitespace().nth(1) {
                    if let Ok(memory_kb) = memory_str.parse::<u64>() {
                        let memory_mb = memory_kb / 1024;
                        info!("{context}: Memory usage: {memory_mb} MB");
                        return;
                    }
                }
            }
        }
    }
    info!("{context}: Memory usage unavailable");
}

#[derive(Debug)]
pub struct AppState {
    pub handle: AppHandle,
}

#[derive(Debug, Clone)]
pub struct QuantizedWeights {
    pub weights: Array2<i8>,
    pub scale: f32,
    pub zero_point: i8,
}

impl QuantizedWeights {
    /// Dequantize a single value from i8 to f32
    pub fn dequantize_value(&self, quantized: i8) -> f32 {
        (quantized as f32 - self.zero_point as f32) * self.scale
    }
    
    /// Dequantize an entire row to f32 for computation
    pub fn dequantize_row(&self, row_idx: usize) -> Vec<f32> {
        if let Some(row) = self.weights.row(row_idx).as_slice() {
            row.iter()
                .map(|&q| self.dequantize_value(q))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Quantize f32 values to i8
    pub fn quantize_value(&self, value: f32) -> i8 {
        let quantized = (value / self.scale) + self.zero_point as f32;
        quantized.round().clamp(i8::MIN as f32, i8::MAX as f32) as i8
    }
}

pub static WEIGHTS: OnceLock<QuantizedWeights> = OnceLock::new();
pub static VOCAB: OnceLock<HashMap<String, usize>> = OnceLock::new();

/// Initializes VOCAB and WEIGHTS to be their respective files
pub fn initialize_globals() {
    info!("Initializing globals");
    log_memory_usage("Before model loading");
    
    WEIGHTS.get_or_init(|| {
        let embedding_dim = get_embedding_dimensions();

        let weights_bytes: Vec<u8> = match fs::read(get_path_to_weights()) {
            Ok(bytes) => bytes,
            Err(e) => {
                error!("Could not read weights: {e}");
                return QuantizedWeights {
                    weights: Array2::zeros((0, 0)),
                    scale: 1.0,
                    zero_point: 0,
                };
            }
        };
        
        log_memory_usage("After reading weights file");
        
        let weights_as_f32: &[f32] = cast_slice(&weights_bytes);

        let vocab_size = if embedding_dim == 0 {
            0
        } else {
            weights_as_f32.len() / embedding_dim
        };

        let quantized_weights = match Array2::from_shape_vec((vocab_size, embedding_dim), weights_as_f32.to_vec()) {
            Ok(arr) => {
                log_memory_usage("After creating f32 array");
                
                // Calculate proper quantization parameters
                let min_val = arr.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = arr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                
                let scale = (max_val - min_val) / (i8::MAX as f32 - i8::MIN as f32);
                let zero_point = ((min_val / scale).round() - i8::MIN as f32) as i8;
                
                info!("Quantization params - Scale: {scale}, Zero point: {zero_point}, Min: {min_val}, Max: {max_val}");
                
                // Quantize the weights
                let quantized_data: Vec<i8> = arr.iter()
                    .map(|&x| {
                        let quantized = (x / scale) + zero_point as f32;
                        quantized.round().clamp(i8::MIN as f32, i8::MAX as f32) as i8
                    })
                    .collect();
                    
                drop(arr); // Free the f32 array memory
                log_memory_usage("After quantizing and dropping f32 array");
                    
                let quantized_array = Array2::from_shape_vec((vocab_size, embedding_dim), quantized_data).unwrap();

                QuantizedWeights {
                    weights: quantized_array,
                    scale,
                    zero_point,
                }
            },
            Err(e) => {
                error!("Shape mismatch in weights: {e}");
                QuantizedWeights {
                    weights: Array2::zeros((0, 0)),
                    scale: 1.0,
                    zero_point: 0,
                }
            }
        };

        quantized_weights
    });

    log_memory_usage("After quantized weights loading");

    VOCAB.get_or_init(|| load_vocab(&get_path_to_vocab()));
    
    log_memory_usage("After vocab loading - initialization complete");
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
    let db_exists = match PathBuf::from(&path).try_exists() {
        Ok(exists) => exists,
        Err(e) => {
            error!("Failed to check db dir existence: {e}");
            false
        }
    };
    if db_exists {
        path.push("files.sqlite3");
        let manager = SqliteConnectionManager::file(path);
        let pool = match Pool::new(manager) {
            Ok(pool) => pool,
            Err(e) => {
                error!("Failed to create pool: {e}");
                panic!("Failed to create pool: {e}");
            }
        };
        if let Ok(conn) = pool.get() {
            initialize_database(&conn);
        } else {
            error!("Initializing failed: could not get connection from pool");
        }
        pool
    } else {
        if let Err(e) = create_dir(PathBuf::from(&path)) {
            error!("Failed to create Dir: {e}");
        }
        path.push("files.sqlite3");
        let manager = SqliteConnectionManager::file(path);
        let pool = match Pool::builder()
            .max_size(5)  // Reduce from default (usually 10)
            .min_idle(Some(1))  // Keep minimum connections
            .build(manager) 
            {
                Ok(pool) => pool,
                Err(e) => {
                    error!("Failed to create pool: {e}");
                    panic!("Failed to create pool: {e}");
                }
            };
        if let Ok(conn) = pool.get() {
            initialize_database(&conn);
        } else {
            error!("Initializing failed: could not get connection from pool");
        }
        pool
    }
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
