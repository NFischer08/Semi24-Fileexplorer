// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
pub mod context_actions;
pub mod database_operations;
pub mod file_information;
pub mod manager;

use file_explorer_lib::rt_db_update::start_file_watcher;
use manager::{manager_check_database, manager_create_database};
use rayon::prelude::*;
use std::collections::HashSet;
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
    println!("Available drives: {:?}", drives);

    thread::spawn(move || {
        drives.into_par_iter().for_each(|drive| {
            manager_create_database(drive).unwrap();
        });
        manager_check_database().unwrap();
    });
    let allowed_file_extensions: HashSet<String> = [
        // Text and documents
        "txt", "pdf", "doc", "docx", "rtf", "odt", "tex", "md", "epub",
        // Spreadsheets and presentations
        "xls", "xlsx", "csv", "ods", "ppt", "pptx", "odp", "key", // Images
        "jpg", "jpeg", "png", "gif", "bmp", "tiff", "svg", "webp", "ico", "raw",
        // Audio and video
        "mp3", "wav", "ogg", "flac", "aac", "wma", "m4a", "mp4", "avi", "mov", "wmv", "flv", "mkv",
        "webm", "m4v", "3gp", // Archives and data
        "zip", "rar", "7z", "tar", "gz", "bz2", "xz", "json", "xml", "yaml", "yml", "toml", "ini",
        "cfg", // Web and programming
        "html", "htm", "css", "js", "php", "asp", "jsp", "py", "java", "c", "cpp", "h", "hpp", "cs",
        "rs", "go", "rb", "pl", "swift", "kt", "ts", "coffee", "scala", "groovy", "lua", "r",
        // Scripts and executables
        "sh", "bash", "zsh", "fish", "bat", "cmd", "ps1", "exe", "dll", "so", "dylib",
        // Other formats
        "sql", "db", "sqlite", "mdb", "ttf", "otf", "woff", "woff2", "obj", "stl", "fbx", "dxf",
        "dwg", "psd", "ai", "ind", "iso", "img", "dmg", "bak", "log", "pcap",
    ]
    .iter()
    .map(|&s| String::from(s))
    .collect();

    thread::spawn(move || start_file_watcher(PathBuf::from(r"C:\"), allowed_file_extensions));

    file_explorer_lib::run();
}
