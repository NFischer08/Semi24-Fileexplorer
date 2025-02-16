use rayon::ThreadPoolBuilder;
use std::{ path::PathBuf, fs::DirEntry };
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use tauri::command;
use crate::database_operations::{check_database, create_database, initialize_database_and_extensions, search_database};
use crate::file_information::{get_file_information, FileType, FileEntry};

#[derive(Debug, serde::Serialize)] // TODO: braucht man diese Zeile?
pub struct SearchResult {
    name: String,
    path: String,
    last_modified: String,
    file_type: String,
    size: String
}

impl SearchResult {
    fn format(file_entry: FileEntry, path: DirEntry) -> SearchResult {
        let (file_type, is_dir) = match file_entry.file_type {
            FileType::Directory => {
                ("Directory".to_string(), true)
            },
            FileType::File(extension) => (extension, false),
            FileType::None => ("File".to_string(), false)
        };
        let size: String = if is_dir {
            "--".to_string()
        }
        else {
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
            path: path.path().to_string_lossy().into_owned(),
            last_modified: file_entry.last_modified.format("%d.%m.%Y %H:%M Uhr").to_string(),
            file_type,
            size
        }
    }
}

fn build_struct(paths: Vec<DirEntry>) -> Vec<SearchResult> {
    paths.into_iter()
        .map(|path| { SearchResult::format(get_file_information(&path), path) }
        )
        .collect()
}

fn manager_make_pooled_connection() -> Result<PooledConnection<SqliteConnectionManager>, Box<dyn std::error::Error>> {
    let manager = SqliteConnectionManager::file("files.sqlite3");
    let connection_pool = Pool::new(manager)?;
    let pooled_connection = connection_pool.get()?;
    pooled_connection.pragma_update(None, "journal_mode", "WAL")?;
    pooled_connection.pragma_update(None, "synchronous", "NORMAL")?;
    pooled_connection.pragma_update(None, "wal_autocheckpoint", "1000")?;
    Ok(pooled_connection)
}

pub fn manager_create_database(
    database_scan_start: PathBuf,
)  -> Result<(), String>
{
    let pooled_connection = match manager_make_pooled_connection() {
        Ok(pooled_connection) => pooled_connection,
        Err(e) => return Err(e.to_string())
    };

    let allowed_file_extensions= match initialize_database_and_extensions(&pooled_connection) {
        Ok(allowed_file_extensions) => allowed_file_extensions,
        Err(e) => return Err(e.to_string())
    };

    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap();



    match create_database(pooled_connection, database_scan_start, &allowed_file_extensions, &thread_pool){
        Ok(_) => {},
        Err(e) => return Err(e.to_string())
    };

    Ok(())
}

#[command]
pub fn manager_basic_search(searchterm: &str, searchpath: &str) -> Result<Vec<SearchResult>, String> {
    let pooled_connection = match manager_make_pooled_connection() {
        Ok(pooled_connection) => pooled_connection,
        Err(e) => return Err(e.to_string())
    };
    let similarity_threshold = 0.7;

    let search_path = PathBuf::from(searchpath);

    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap();

    println!("{}", searchpath);

    let return_paths = match search_database(&pooled_connection, searchterm, similarity_threshold, &thread_pool, search_path) {
        Ok(return_paths) => return_paths,
        Err(e) => return Err(e.to_string())
    };  // Hier kann das Frontend abgreifen

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