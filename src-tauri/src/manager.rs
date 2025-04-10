use crate::db_create::create_database;
use crate::db_search::search_database;
use crate::db_util::{initialize_database, load_vocab};
use crate::file_information::{get_file_information, FileData, FileDataFormatted};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::HashMap;
use std::env;
use std::fs::create_dir;
use std::path::absolute;
use std::sync::LazyLock;
use std::{fs::DirEntry, path::PathBuf};
use tauri::command;
use tch::CModule;

pub static CURRENT_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    env::current_dir()
        .and_then(|cwd| absolute(cwd))
        .expect("Failed to resolve absolute path")
    // VERY IMPORTANT when using .push() don't start with a /, if you do it will override the path with C: + "Your Input"
});

pub static THREAD_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    println!("Thread pool built");
    ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap()
});

pub static MODEL: LazyLock<CModule> = LazyLock::new(|| {
    let mut path = CURRENT_DIR.clone();
    path.push("data/model/model.pt");
    CModule::load(path).expect("Unable to load model")
});

pub static VOCAB: LazyLock<HashMap<String, usize>> = LazyLock::new(|| {
    let mut path = CURRENT_DIR.clone();
    path.push("data/model/vocab.json");
    load_vocab(&path)
});

fn build_struct(paths: Vec<DirEntry>) -> Vec<FileDataFormatted> {
    paths
        .into_iter()
        .map(|path| FileData::format(get_file_information(&path)))
        .collect()
}

fn manager_make_pooled_connection() -> Pool<SqliteConnectionManager> {
    let mut path = CURRENT_DIR.clone();
    path.push("data/db");
    if PathBuf::from(&path).try_exists().expect("Reason") {
        path.push("files.sqlite3");
        let manager = SqliteConnectionManager::file(PathBuf::from(path));
        let connection_pool = Pool::new(manager).expect("Failed to create pool.");
        connection_pool
    } else {
        create_dir(PathBuf::from(&path)).expect("Failed to create Dir");
        path.push("files.sqlite3");
        let manager = SqliteConnectionManager::file(PathBuf::from(path));
        let connection_pool = Pool::new(manager).expect("Failed to create pool.");
        connection_pool
    }
}

pub fn manager_create_database(database_scan_start: PathBuf) -> Result<(), String> {
    let connection_pool = manager_make_pooled_connection();

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

// searchfiletype is the Filetype Ending without the Dot, for Directorys it must be dir
#[command]
pub fn manager_basic_search(
    searchterm: &str,
    searchpath: &str,
    searchfiletype: &str,
) -> Result<Vec<FileDataFormatted>, String> {
    let connection_pool = manager_make_pooled_connection();

    let number_results_embedding = 30;
    let number_results_levenhstein = 10;

    let search_path = PathBuf::from(searchpath);

    let return_paths = match search_database(
        connection_pool,
        searchterm,
        search_path,
        searchfiletype,
        &MODEL,
        number_results_embedding,
        number_results_levenhstein,
    ) {
        Ok(return_paths) => return_paths,
        Err(e) => return Err(e.to_string()),
    };

    let search_result = build_struct(return_paths);

    Ok(search_result)
}
