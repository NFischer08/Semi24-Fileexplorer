// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
pub mod config_handler;
pub mod context_actions;
pub mod db_create;
pub mod db_search;
pub mod db_util;
pub mod file_information;
pub mod manager;
pub mod rt_db_update;

use crate::config_handler::{get_number_of_threads, get_paths_to_index, CURRENT_DIR};
use crate::manager::{initialize_globals, manager_populate_database, AppState};
use crate::rt_db_update::start_file_watcher;
use config_handler::{get_css_settings, get_fav_file_extensions, initialize_config};
use context_actions::{copy_file, cut_file, delete_file, open_file, paste_file, rename_file};
use file_information::format_file_data;
use manager::manager_basic_search;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{fs::create_dir, thread};
use tauri::Manager;

fn setup_directory_structure() {
    let data_dir = CURRENT_DIR.join("data");
    let model_dir = data_dir.join("model");
    let config_dir = data_dir.join("config");
    let tmp_dir = data_dir.join("tmp");

    match (
        data_dir.exists(),
        model_dir.exists(),
        config_dir.exists(),
        tmp_dir.exists(),
    ) {
        (false, _, _, _) => {
            create_dir(&data_dir).expect("Unable to create data directory");
            create_dir(&model_dir).expect("Unable to create model directory");
            create_dir(&config_dir).expect("Unable to create config directory");
            create_dir(&tmp_dir).expect("Unable to create tmp directory");
        }
        (true, false, _, _) => create_dir(&model_dir).expect("Unable to create model directory"),
        (true, _, false, _) => create_dir(&config_dir).expect("Unable to create config directory"),
        (true, _, _, false) => create_dir(&tmp_dir).expect("Unable to create tmp directory"),
        _ => {}
    }
}

pub fn run() {
    initialize_config();

    rayon::ThreadPoolBuilder::new()
        .num_threads(get_number_of_threads()) // Reserve one core for OS
        .build_global()
        .expect("Couldn't build thread pool");

    setup_directory_structure();

    let paths_to_index = get_paths_to_index();
    thread::spawn(move || {
        paths_to_index.par_iter().for_each(|path| {
            manager_populate_database(path.clone()).unwrap();
        });
    });

    tauri::Builder::default()
        .setup(|app| {
            app.manage(AppState {
                handle: app.handle().clone(),
            });
            initialize_globals();
            thread::spawn(move || start_file_watcher());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            format_file_data,
            copy_file,
            paste_file,
            cut_file,
            delete_file,
            rename_file,
            open_file,
            manager_basic_search,
            get_fav_file_extensions,
            get_css_settings,
        ])
        .run(tauri::generate_context!("tauri.conf.json"))
        .expect("error while running tauri application");
}
