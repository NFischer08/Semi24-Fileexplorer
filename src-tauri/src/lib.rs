// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
pub mod config_handler;
pub mod context_actions;
pub mod db_create;
pub mod db_search;
pub mod db_util;
pub mod file_information;
pub mod manager;
pub mod rt_db_update;

use crate::config_handler::{
    build_config, get_number_of_threads, get_paths_to_index, ColorConfig, Settings, CURRENT_DIR,
};
use crate::manager::{
    check_for_default_paths, initialize_globals, manager_populate_database, AppState,
};
use crate::rt_db_update::start_file_watcher;
use config_handler::{get_css_settings, get_fav_file_extensions, initialize_config};
use context_actions::{copy_file, cut_file, delete_file, open_file, paste_file, rename_file};
use file_information::format_file_data;
use log::{error, warn, LevelFilter};
use manager::manager_basic_search;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{fs::create_dir, thread};
use tauri::Manager;

fn setup_directory_structure() {
    let data_dir = CURRENT_DIR.join("data");
    let model_dir = data_dir.join("model");
    let config_dir = data_dir.join("config");
    let config_file = config_dir.join("config.json");
    let color_config_file = config_dir.join("color-config.json");
    let tmp_dir = data_dir.join("tmp");

    match (
        data_dir.exists(),
        model_dir.exists(),
        config_dir.exists(),
        tmp_dir.exists(),
    ) {
        (false, _, _, _) => {
            if let Err(e) = create_dir(&data_dir) {
                error!("Unable to create data directory: {}", e);
            }
            if let Err(e) = create_dir(&model_dir) {
                error!("Unable to create model directory: {}", e);
            }
            if let Err(e) = create_dir(&config_dir) {
                error!("Unable to create config directory: {}", e);
            }
            if let Err(e) = create_dir(&tmp_dir) {
                error!("Unable to create tmp directory: {}", e);
            }
        }
        (true, false, _, _) => {
            if let Err(e) = create_dir(&model_dir) {
                error!("Unable to create model directory: {}", e);
            }
        }
        (true, _, false, _) => {
            if let Err(e) = create_dir(&config_dir) {
                error!("Unable to create config directory: {}", e);
            }
        }
        (true, _, _, false) => {
            if let Err(e) = create_dir(&tmp_dir) {
                error!("Unable to create tmp directory: {}", e);
            }
        }
        _ => {}
    }

    if !config_file.exists() {
        if build_config(&config_file, &Settings::default()) {
            warn!("Config file didn't exist, created new one");
        } else {
            warn!("Unable to build config file");
        }
    }
    if !color_config_file.exists() {
        if build_config(&color_config_file, &ColorConfig::default()) {
            warn!("Color-config file didn't exist, created new one");
        } else {
            warn!("Unable to build color-config file");
        }
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::new()
                .target(tauri_plugin_log::Target::new(
                    tauri_plugin_log::TargetKind::Folder {
                        path: std::path::PathBuf::from(CURRENT_DIR.clone()),
                        file_name: None,
                    },
                ))
                .level(LevelFilter::Warn)
                .build(),
        )
        .setup(move |app| {
            check_for_default_paths();

            setup_directory_structure();

            initialize_config();
            initialize_globals();

            let thread_pool_result = rayon::ThreadPoolBuilder::new()
                .panic_handler(|err| {
                    error!("A Rayon worker thread panicked: {:?}", err);
                })
                .num_threads(get_number_of_threads()) // Reserve one core for OS
                .build_global();

            if let Err(e) = thread_pool_result {
                error!("Couldn't build thread pool: {}", e);
            }

            let paths_to_index = get_paths_to_index();
            thread::spawn(move || {
                paths_to_index.par_iter().for_each(|path| {
                    if let Err(e) = manager_populate_database(path.clone()) {
                        error!("Failed to populate database for {}: {}", path.display(), e);
                    }
                });
            });

            app.manage(AppState {
                handle: app.handle().clone(),
            });
            thread::spawn(start_file_watcher);
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
        .unwrap_or_else(|e| error!("Error while running tauri application: {}", e));
}
