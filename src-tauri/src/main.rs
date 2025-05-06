// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
pub mod config_handler;
pub mod context_actions;
pub mod db_create;
pub mod db_search;
pub mod db_util;
pub mod file_information;
pub mod manager;
pub mod rt_db_update;

use crate::config_handler::{get_number_of_threads, get_paths_to_index};
use crate::manager::initialize_globals;
use file_explorer_lib::manager::{manager_create_database, CURRENT_DIR};
use file_explorer_lib::rt_db_update::start_file_watcher;
use rayon::prelude::*;
use std::fs::create_dir;
use std::thread;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(get_number_of_threads()) // Reserve one core for OS
        .build_global()
        .expect("Couldn't build thread pool");

    //TODO Schönes Match statement bitte Nino
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

    let paths_to_index = get_paths_to_index();

    thread::spawn(move || {
        paths_to_index.par_iter().for_each(|path| {
            manager_create_database(path.clone()).unwrap();
        });
    });
    thread::spawn(move || start_file_watcher());

    file_explorer_lib::run();
}
