use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;
use chrono::{Local, TimeZone};
use crate::{FileEntry, FileType, FileDataFormatted};
use tauri::command;
use fs::DirEntry;

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
                        entries.push(get_file_information(entry))
                    }
                    Err(e) => return Err(e.to_string()),
                }
            }
        }
        Err(e) => return Err(e.to_string()),
    }

    Ok(entries)
}

fn get_file_information(entry: DirEntry) -> FileEntry{
    // get the name of the file
    let file_name = entry.file_name().into_string().unwrap_or_default();

    // get the metadata of the file
    let metadata = match entry.metadata() {
        Ok(metadata) => metadata,
        Err(_) => { // if an Error occurs while catching metadata, the name gets return and the other values are set to the standard
            return FileEntry {
                name: file_name,
                file_type: FileType::None,
                last_modified: Local::now(),
                size_in_kb: 0
            }
        },
    };

    // get the filetype of file
    let file_type = if metadata.is_dir() {
        FileType::Directory // either type directory or ...
    } else {
        match entry.path().extension() {
            Some(ext) => FileType::File(ext.to_string_lossy().into_owned()), // ... type actual file (-> extension as String) or ...
            None => FileType::None // ... no file extension at all
        }
    };

    // size of the file in KB, if folder: 0
    let size: u64 = metadata.len() / 1000;

    // get the last modified time of the file
    let modified_time = match metadata.modified() {
        Ok(time) => time,
        Err(_) => { // if it's unable to read the modified time it returns all information currently known
            return FileEntry {
                name: file_name,
                last_modified: Local::now(), // last_modified is set to the current time
                file_type,
                size_in_kb: size
            }
        }
    }; // Convert the last modified time to a readable format
    // Convert SystemTime to DateTime<Local>
    let last_modified = modified_time.duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| Local.timestamp_opt(d.as_secs() as i64, d.subsec_nanos())
            .single()
            .unwrap_or_else(Local::now)
        )        // Fallback to current time if there's an error
        .unwrap(); // TODO => keine Ahnung was das macht!

    // append the important information to the Vector with the FileEntries
    FileEntry {
        name: file_name,
        last_modified,
        file_type,
        size_in_kb: size
    }
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
                    "--".to_string()
                }
                else {
                    let size_kb_f: f64 = file.size_in_kb as f64;
                    let (size, unit) = if file.size_in_kb < 1000 {
                        (size_kb_f, "KB")
                    } else if file.size_in_kb < 1024 * 1024 {
                        (size_kb_f / 1_000.0, "MB")
                    } else if file.size_in_kb < 1024 * 1024 * 1024 {
                        (size_kb_f / 1_000_000.0, "GB")
                    } else {
                        (size_kb_f / 1_000_000_000.0, "TB")
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