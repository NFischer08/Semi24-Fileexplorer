use crate::db_create::create_database;
use crate::db_search::search_database;
use crate::db_util::{get_allowed_file_extensions, initialize_database};
use crate::file_information::{get_file_information, FileEntry, FileType};
use once_cell::sync::Lazy;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{fs::DirEntry, path::PathBuf};
use std::cell::RefCell;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use tauri::command;
use std::time::Instant;
use tch::{CModule};

#[derive(serde::Serialize)]
pub struct SearchResult {
    name: String,
    path: String,
    last_modified: String,
    file_type: String,
    size: String,
}

pub static THREAD_POOL: Lazy<ThreadPool> = Lazy::new(|| {
    ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap()
});
pub static MODEL: Lazy<TextEmbedding> = Lazy::new(|| {
    println!("Initializing TextEmbedding model...");
    let model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::MultilingualE5Small)
            .with_show_download_progress(true),
    )
        .expect("Could not create TextEmbedding");
    model
});

thread_local! {
    pub static LOCAL_MODEL: RefCell<CModule> = RefCell::new(
        CModule::load("src-tauri/src/neural_network/skipgram_model_script.pt").expect("Failed to load model")
    );
}

impl SearchResult {
    fn format(file_entry: FileEntry, path: DirEntry) -> SearchResult {
        let (file_type, is_dir) = match file_entry.file_type {
            FileType::Directory => ("Directory".to_string(), true),
            FileType::File(extension) => (extension, false),
            FileType::None => ("File".to_string(), false),
        };
        let size: String = if is_dir {
            "--".to_string()
        } else {
            let size_kb_f: f64 = file_entry.size_in_kb as f64;
            let (size, unit) = if file_entry.size_in_kb < 1000 {
                (size_kb_f, "KB")
            } else if file_entry.size_in_kb < 1_000_000 {
                (size_kb_f / 1_000.0, "MB")
            } else if file_entry.size_in_kb < 1_000_000_000 {
                (size_kb_f / 1_000_000.0, "GB")
            } else {
                (size_kb_f / 1_000_000_000.0, "TB")
            };

            // Round to one decimal place
            let rounded_size = (size * 10.0).round() / 10.0;

            // Format the output
            format!("{:.1} {}", rounded_size, unit)
        };

        SearchResult {
            name: file_entry.name,
            path: path.path().to_string_lossy().replace("\\", "/"), // replace backslashes with slashes since DirEntry loves \ on Windows
            last_modified: file_entry
                .last_modified
                .format("%d.%m.%Y %H:%M Uhr")
                .to_string(),
            file_type,
            size,
        }
    }
}

fn build_struct(paths: Vec<DirEntry>) -> Vec<SearchResult> {
    paths
        .into_iter()
        .map(|path| SearchResult::format(get_file_information(&path), path))
        .collect()
}

fn manager_make_pooled_connection(
) -> Result<Pool<SqliteConnectionManager>, Box<dyn std::error::Error>> {
    let manager = SqliteConnectionManager::file("files.sqlite3");
    let connection_pool = Pool::new(manager).expect("Failed to create pool.");
    Ok(connection_pool)
}

pub fn manager_create_database(database_scan_start: PathBuf) -> Result<(), String> {
        let connection_pool = match manager_make_pooled_connection() {
        Ok(connection_pool) => connection_pool,
        Err(e) => return Err(e.to_string()),
    };

    initialize_database(&connection_pool.get().expect("Initializing failed: "));

    let allowed_file_extensions = get_allowed_file_extensions();

    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap();

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

    match create_database(
        connection_pool,
        database_scan_start,
        &allowed_file_extensions,
        &thread_pool,
        &MODEL,
        "src-tauri/src/neural_network/skipgram_model_script.pt",
    ) {
        Ok(_) => {}
        Err(e) => return Err(e.to_string()),
    };

    Ok(())
}

// searchfiletype is the Filetype Ending without the Dot, for Directorys it must be dir
#[command]
pub fn manager_basic_search(
    searchterm: &str,
    searchpath: &str,
    searchfiletype: &str,
) -> Result<Vec<SearchResult>, String> {
    println!("manager basic search");

    let manager_search_time = Instant::now();
    
    let connection_pool = match manager_make_pooled_connection() {
        Ok(connection_pool) => connection_pool,
        Err(e) => return Err(e.to_string()),
    };

    let number_results = 20;


    let search_path = PathBuf::from(searchpath);
    println!("Manager before FN took: {}", manager_search_time.elapsed().as_millis());


    let return_paths = match search_database(
        connection_pool,
        searchterm,
        &THREAD_POOL,
        search_path,
        searchfiletype,
        &MODEL,
        number_results
    ) {
        Ok(return_paths) => return_paths,
        Err(e) => return Err(e.to_string()),
    };
    println!("Manager Return Paths took: {}", manager_search_time.elapsed().as_millis());


    let search_result = build_struct(return_paths);
    println!("Manager Search took: {}", manager_search_time.elapsed().as_millis());
    
    Ok(search_result)
}