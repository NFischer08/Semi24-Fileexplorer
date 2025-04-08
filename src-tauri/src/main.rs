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
use file_explorer_lib::manager::CURRENT_DIR;

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


    if !PathBuf::from(&*CURRENT_DIR).join("data").exists() {
        create_dir(&*CURRENT_DIR.join("data"))
            .expect("Unable to create data directory");
        create_dir(&*CURRENT_DIR.join("data/model"))
            .expect("Unable to create model directory");
        create_dir(&*CURRENT_DIR.join("data/config"))
            .expect("Unable to create config directory");
    };

    let mut drives = get_all_drives();
    println!("Available drives: {:?}", drives);

    match initialize_config() {
        Ok(x) => {println!("{}", x)}
        Err(e) => panic!("Failed to initialize config: {e}"),
    }

    thread::spawn(move || {
        drives.into_par_iter().for_each(|drive| {
            manager_create_database(drive).unwrap();
        });
    });

    file_explorer_lib::run();

}
