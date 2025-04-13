use crate::db_create::create_database;
use crate::db_search::search_database;
use crate::db_util::{initialize_database, load_vocab};
use crate::file_information::{get_file_information, FileData, FileDataFormatted};
use crate::config_handler::{get_number_results_levenhstein, get_number_results_embedding};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::HashMap;
use std::env;
use std::fs::create_dir;
use std::path::absolute;
use std::sync::LazyLock;
use std::{fs::DirEntry, path::PathBuf};
use tauri::{command, Emitter};
use tch::CModule;
use tauri::{State, AppHandle};


#[derive(Debug)]
pub(crate) struct AppState {
    pub(crate) handle: AppHandle,
}

pub static CURRENT_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    env::current_dir()
        .and_then(absolute)
        .expect("Failed to resolve absolute path")
    // VERY IMPORTANT when using .push() don't start with a /, if you do it will override the path with C: + "Your Input"
});

pub static THREAD_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap()
});

pub static MODEL: LazyLock<CModule> = LazyLock::new(|| {
    let mut path = CURRENT_DIR.clone();
    path.push("data/model/model.pt");
    println!("{}", path.display());
    CModule::load(path).expect("Unable to load model")
});

pub static VOCAB: LazyLock<HashMap<String, usize>> = LazyLock::new(|| {
    let mut path = CURRENT_DIR.clone();
    path.push("data/model/vocab.json");
    println!("{}", path.display());
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
        let manager = SqliteConnectionManager::file(path);
        let connection_pool = Pool::new(manager).expect("Failed to create pool.");
        connection_pool
    } else {
        create_dir(PathBuf::from(&path)).expect("Failed to create Dir");
        path.push("files.sqlite3");
        let manager = SqliteConnectionManager::file(path);
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
#[command(async)]
pub fn manager_basic_search(
    searchterm: &str,
    searchpath: &str,
    searchfiletype: &str,
    state: State<AppState>
) -> Result<(), String>   {
    let connection_pool = manager_make_pooled_connection();

    let search_path = PathBuf::from(searchpath);

    let return_paths = match search_database(
        connection_pool,
        searchterm,
        search_path,
        searchfiletype,
        &MODEL,
        get_number_results_embedding(),
        get_number_results_levenhstein(),
    ) {
        Ok(return_paths) => return_paths,
        Err(e) => return Err(e.to_string()),
    };

    let search_result = build_struct(return_paths);

    emit_search(&state.handle, search_result);
    Ok(())
}

fn emit_search(app: &AppHandle, search_results: Vec<FileDataFormatted>) {
    println!("I have somethin' for ya'll");
    app.emit("search_finished", &search_results).unwrap();
    println!("I had somethin' for ya'll");
}