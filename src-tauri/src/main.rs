// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
pub mod config_handler;
pub mod context_actions;
pub mod db_create;
pub mod db_search;
pub mod db_util;
pub mod file_information;
pub mod manager;

use crate::config_handler::get_number_of_threads;
use crate::manager::initialize_globals;
use file_explorer_lib::manager::{manager_create_database, CURRENT_DIR};
use file_explorer_lib::rt_db_update::start_file_watcher;
use rayon::prelude::*;
use std::fs::create_dir;
use std::path::PathBuf;
use std::thread;

fn get_all_drives() -> Vec<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        use windows::Win32::Storage::FileSystem::GetLogicalDrives;
        let drives = unsafe { GetLogicalDrives() };
        (0..26)
            .filter_map(|i| {
                if (drives & (1 << i)) != 0 {
                    Some(PathBuf::from(format!("{}:\\", (b'A' + i) as char)))
                } else {
                    None
                }
            })
            .collect()
    }

    #[cfg(target_os = "linux")]
    {
        vec![PathBuf::from("/")]
    }

    #[cfg(target_os = "macos")]
    {
        use std::fs;
        fs::read_dir("/Volumes")
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .collect()
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        vec![PathBuf::from("/")]
    }
}

fn main() {
    initialize_globals();
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

    let drives = get_all_drives();
    /*
    let mut drives: Vec<PathBuf> = Vec::new();
    drives.push(PathBuf::from(r"C:\Users\maxmu"));

     */

    thread::spawn(move || {
        drives.par_iter().for_each(|drive| {
            manager_create_database(drive.clone()).unwrap();
        });
    });
    thread::spawn(move || start_file_watcher());

    file_explorer_lib::run();
}
