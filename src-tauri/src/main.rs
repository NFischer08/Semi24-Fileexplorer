// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
pub mod context_actions;
pub mod db_create;
pub mod db_search;
pub mod db_util;
pub mod file_information;
pub mod manager;
pub mod config_handler;

use config_handler::initialize_config;
use manager::manager_create_database;
use rayon::prelude::*;
use std::fs::{create_dir};
use rayon::prelude::*;
use std::path::PathBuf;
use std::thread;
use rayon::prelude::*;
use file_explorer_lib::manager::{CURRENT_DIR};

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
    println!("{}", num_cpus::get());

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get() - 1) // Reserve one core for OS
        .build_global()
        .expect("Couldn't build thread pool");

    tch::set_num_interop_threads(1);
    tch::set_num_threads((num_cpus::get() - 1) as i32);


    //TODO Schönes Match statement bitte Nino
    let data_dir = CURRENT_DIR.join("data");
    let model_dir = data_dir.join("model");
    let config_dir = data_dir.join("config");
    let tmp_dir = data_dir.join("tmp");

    match (data_dir.exists(), model_dir.exists(), config_dir.exists(), tmp_dir.exists()) {
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


    let mut drives = get_all_drives();
    println!("Available drives: {:?}", drives);

    match initialize_config() {
        Ok(x) => {println!("{}", x)}
        Err(e) => panic!("Failed to initialize config: {e}"),
    }

    thread::spawn(move || {
        drives.par_iter().for_each(|drive| {
            manager_create_database(drive.clone()).unwrap();
        });
    });

    file_explorer_lib::run();

}
