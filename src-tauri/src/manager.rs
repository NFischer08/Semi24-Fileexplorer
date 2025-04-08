use crate::file_information::{get_file_information, FileData, FileDataFormatted};
use crate::db_create::create_database;
use crate::db_search::search_database;
use crate::db_util::{get_allowed_file_extensions, initialize_database, load_vocab};
use once_cell::sync::Lazy;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{fs::DirEntry, path::PathBuf};
use std::collections::HashMap;
use std::fs::create_dir;
use tauri::command;
use tch::CModule;
use std::env;
use std::path::{absolute};

pub static CURRENT_DIR: Lazy<PathBuf> = Lazy::new(|| {
     env::current_dir()
        .and_then(|cwd| absolute(cwd))
        .expect("Failed to resolve absolute path")
    // VERY IMPORTANT when using .push() don't start with a /, if you do it will override the path with C: + "Your Input"
});


pub static THREAD_POOL: Lazy<ThreadPool> = Lazy::new(|| {
    ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap()
});

fn build_struct(paths: Vec<DirEntry>) -> Vec<FileDataFormatted> {
    paths
        .into_iter()
        .map(|path| FileData::format(get_file_information(&path)))
        .collect()
}

pub static MODEL: Lazy<CModule> = Lazy::new(|| {
    let mut path = CURRENT_DIR.clone();
    path.push("data/model/model.pt");
    println!("Model Path : {}", path.display());
    CModule::load(path).expect("Unable to load model")
});

pub static VOCAB: Lazy<HashMap<String, usize>> = Lazy::new(|| {
    let mut path = CURRENT_DIR.clone();
    path.push("data/model/vocab.json");
    load_vocab(&path)
});

fn manager_make_pooled_connection(
) -> Pool<SqliteConnectionManager> {
    let mut path = CURRENT_DIR.clone();
    path.push("data/db");
    println!("CURRENT_DIR = {}", path.to_str().unwrap());
    if PathBuf::from(&path).try_exists().expect("Reason") {
        path.push("files.sqlite3");
        let manager = SqliteConnectionManager::file(PathBuf::from(path));
        let connection_pool = Pool::new(manager).expect("Failed to create pool.");
        connection_pool
    }
    else {
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

    let pymodel = "./data/model/model.pt";

    let database_scan_start = PathBuf::from(database_scan_start);

    match create_database(
        connection_pool,
        database_scan_start,
        &allowed_file_extensions,
        &thread_pool,
        &pymodel,
        &VOCAB,
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
) -> Result<Vec<FileDataFormatted>, String> {
    let connection_pool = manager_make_pooled_connection();

    let number_results_embedding = 50;
    let number_results_levenhstein = 5;

    let search_path = PathBuf::from(searchpath);

    let return_paths = match search_database(
        connection_pool,
        searchterm,
        &THREAD_POOL,
        search_path,
        searchfiletype,
        &MODEL,
        number_results_embedding,
        number_results_levenhstein,
        &VOCAB
    ) {
        Ok(return_paths) => return_paths,
        Err(e) => return Err(e.to_string())
    };

    let search_result = build_struct(return_paths);

    Ok(search_result)
}
