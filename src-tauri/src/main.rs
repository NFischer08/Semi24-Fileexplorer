// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
pub mod context_actions;
pub mod db_create;
pub mod db_search;
pub mod db_util;
pub mod file_information;
pub mod manager;

use manager::{manager_create_database};
use rayon::prelude::*;
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
    let mut drives = get_all_drives();
    drives.clear();
    drives.push(PathBuf::from(r"C:\Users\maxmu"));
    println!("Available drives: {:?}", drives);
    thread::spawn(move || {
        drives.into_par_iter().for_each(|drive| {
            manager_create_database(drive).unwrap();
        });
    });

    file_explorer_lib::run();
}
