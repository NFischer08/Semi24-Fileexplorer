// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/

use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;
use tauri::command;
use chrono::{DateTime, Local, TimeZone};

#[derive(Debug, serde::Serialize)]
struct FileEntry {
    name: String,
    last_modified: String,
    file_type: String,
    size: u64
}

#[command]
fn list_files_and_folders(path: String) -> Result<Vec<FileEntry>, String> {
    let path = PathBuf::from(path);

    // Check if the path exists and is a directory
    if !path.exists() || !path.is_dir() {
        return Err("The specified path does not exist or is not a directory.".into());
    }

    let mut entries: Vec<FileEntry> = Vec::new();

    // Read the directory entries
    match fs::read_dir(&path) {
        Ok(dir_entries) => {
            for entry in dir_entries {
                match entry {
                    Ok(entry) => {
                        let file_name = entry.file_name().into_string().unwrap_or_default();
                        let metadata = entry.metadata().map_err(|e| e.to_string())?;
                        let modified_time = metadata.modified().map_err(|e| e.to_string())?;
                        let file_type = if metadata.is_dir() {
                            "Directory".to_string()
                        } else {
                            match entry.path().extension() {
                                Some(ext) => ext.to_string_lossy().into_owned(),
                                None => "Unknown".to_string(),
                            }
                        };
                        let size: u64 = metadata.len() / 1000; // size of the file in KB, if folder: 0

                        // Convert the last modified time to a readable format
                        let last_modified = format_time(modified_time);

                        entries.push(FileEntry {
                            name: file_name,
                            last_modified,
                            file_type,
                            size
                        });
                    }
                    Err(e) => return Err(e.to_string()),
                }
            }
        }
        Err(e) => return Err(e.to_string()),
    }

    Ok(entries)
}

// Function to format the last modified time
fn format_time(time: SystemTime) -> String {
    // Convert SystemTime to DateTime<Local>
    let datetime: DateTime<Local> = time
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| Local.timestamp(d.as_secs() as i64, d.subsec_nanos()))
        .unwrap_or_else(|_| Local::now()); // Fallback to current time if there's an error

    datetime.format("%d.%m.%Y %H:%M Uhr").to_string()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![list_files_and_folders])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
