use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;
use chrono::{DateTime, Local, TimeZone};
use crate::{FileEntry, FileType, FileDataFormatted};
use tauri::command;

fn list_files_and_folders(path: &str) -> Result<Vec<FileEntry>, String> {
    let path = PathBuf::from(path);

    // Check if the path exists
    if !path.exists() {
        return Err("The specified path does not exist.".into());
    }

    // Check if the path is a directory
    if !path.is_dir() {
        return Err("The specified path is not a directory.".into());
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
                            FileType::Directory
                        } else {
                            match entry.path().extension() {
                                Some(ext) => FileType::File(ext.to_string_lossy().into_owned()),
                                None => FileType::None
                            }
                        };
                        let size: u64 = metadata.len() / 1024; // size of the file in KB, if folder: 0

                        // Convert the last modified time to a readable format
                        let last_modified = get_last_modified_time(modified_time);

                        entries.push(FileEntry {
                            name: file_name,
                            last_modified,
                            file_type,
                            size_in_kb: size
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
fn get_last_modified_time(time: SystemTime) -> DateTime<Local> { // String
    // Convert SystemTime to DateTime<Local>
    time.duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| Local.timestamp_opt(d.as_secs() as i64, d.subsec_nanos())
            .single()
            .unwrap_or_else(Local::now)
        )        // Fallback to current time if there's an error
        .unwrap()
}

#[command]
pub fn format_file_data(path: &str) -> Result<Vec<FileDataFormatted>, String> {
    let files = list_files_and_folders(path);

    match files {
        Ok(files) => {
            let mut formatted_files: Vec<FileDataFormatted> = Vec::new();

            for file in files {
                let (file_type, is_dir) = match file.file_type {
                    FileType::Directory => {
                        ("Directory".to_string(), true)
                    },
                    FileType::File(extension) => (extension, false),
                    FileType::None => ("File".to_string(), false)
                };
                let size: String = if is_dir {
                    "Unknown".to_string()
                }
                else {
                    let size_kb_f: f64 = file.size_in_kb as f64;
                    let (size, unit) = if file.size_in_kb < 1024 {
                        (size_kb_f, "KB")
                    } else if file.size_in_kb < 1024 * 1024 {
                        (size_kb_f / 1024.0, "MB")
                    } else if file.size_in_kb < 1024 * 1024 * 1024 {
                        (size_kb_f / (1024.0 * 1024.0), "GB")
                    } else {
                        (size_kb_f / (1024.0 * 1024.0 * 1024.0), "TB")
                    };

                    // Round to one decimal place
                    let rounded_size = (size * 10.0).round() / 10.0;

                    // Format the output
                    format!("{:.1} {}", rounded_size, unit)
                };

                formatted_files.push(FileDataFormatted {
                    name: file.name,
                    last_modified: file.last_modified.format("%d.%m.%Y %H:%M Uhr").to_string(),
                    file_type,
                    size
                })
            }
            Ok(formatted_files)
        }
        Err(error) => Err(error)
    }
}