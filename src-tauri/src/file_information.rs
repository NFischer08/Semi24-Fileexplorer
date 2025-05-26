use crate::context_actions::clean_path;
use chrono::{DateTime, Local, TimeZone};
use log::{error, info, warn};
use std::{
    fs::{self, DirEntry},
    path::PathBuf,
    time::SystemTime,
};
use tauri::command; // <-- Add logging macros

// custom enum to handle all possible file types
#[derive(Debug)]
pub(crate) enum FileType {
    Directory,
    File(String),
    None,
}

#[derive(Debug)]
pub struct FileData {
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) last_modified: DateTime<Local>,
    pub(crate) file_type: FileType,
    pub(crate) size_in_kb: u64,
}

#[derive(Debug, serde::Serialize, Clone)]
pub struct FileDataFormatted {
    name: String,
    path: String,
    last_modified: String,
    file_type: String,
    size: String,
}

impl FileData {
    pub fn format(self) -> FileDataFormatted {
        let (file_type, is_dir) = match self.file_type {
            FileType::Directory => ("Directory".to_string(), true),
            FileType::File(extension) => (extension, false),
            FileType::None => ("Binary".to_string(), false),
        };
        let size: String = if is_dir {
            "--".to_string()
        } else {
            let size_kb_f: f64 = self.size_in_kb as f64;
            let (size, unit) = if self.size_in_kb < 1_024 {
                (size_kb_f, "KiB")
            } else if self.size_in_kb < 1_048_576 {
                (size_kb_f / 1_024.0, "MiB")
            } else if self.size_in_kb < 1_073_741_824 {
                (size_kb_f / 1_048_576.0, "GiB")
            } else {
                (size_kb_f / 1_073_741_824.0, "TiB")
            };

            // Round to one decimal place
            let rounded_size = (size * 10.0).round() / 10.0;

            // Format the output
            format!("{:.1} {}", rounded_size, unit)
        };
        FileDataFormatted {
            name: self.name,
            path: self.path.to_string_lossy().to_string().replace("\\", "/"),
            last_modified: self.last_modified.format("%d.%m.%Y %H:%M").to_string(),
            file_type,
            size,
        }
    }
}

impl Default for FileData {
    fn default() -> FileData {
        FileData {
            name: String::from("No Name"),
            path: PathBuf::from("/"),
            last_modified: Local::now(),
            file_type: FileType::None,
            size_in_kb: 0,
        }
    }
}

fn list_files_and_folders(filepath: String) -> Result<Vec<FileData>, String> {
    let path = clean_path(filepath);

    // Check if the path exists
    if !path.exists() {
        return Err("The specified path does not exist.".into());
    }

    // Check if the path is a directory
    if !path.is_dir() {
        return Err("The specified path is not a directory.".into());
    }

    if !path.to_string_lossy().to_string().contains("/") {
        warn!(
            "The specified path '{}' is not valid. Seems like you forgot a slash.",
            path.to_string_lossy()
        );
        return Err("The specified path is not valid. Seems like you forgot a slash.".into());
    }

    let mut entries: Vec<FileData> = Vec::new();

    // Read the directory entries
    match fs::read_dir(&path) {
        Ok(dir_entries) => {
            for entry in dir_entries {
                match entry {
                    Ok(entry) => entries.push(get_file_information(&entry)),
                    Err(e) => {
                        error!("Failed to read directory entry: {}", e);
                        return Err(e.to_string());
                    }
                }
            }
        }
        Err(e) => {
            error!("Failed to read directory '{}': {}", path.display(), e);
            return Err(e.to_string());
        }
    }

    Ok(entries)
}

pub fn get_file_information(entry: &DirEntry) -> FileData {
    // get the name of the file
    let file_name = entry.file_name().into_string().unwrap_or_default();
    let path = entry.path();

    // get the metadata of the file
    let metadata = match entry.metadata() {
        Ok(metadata) => metadata,
        Err(e) => {
            warn!("Failed to get metadata for '{}': {}", path.display(), e);
            // if an Error occurs while catching metadata, the name gets return and the other values are set to the standard
            return FileData {
                name: file_name,
                path,
                file_type: FileType::None,
                last_modified: Local::now(),
                size_in_kb: 0,
            };
        }
    };

    // get the filetype of a file
    let file_type = if metadata.is_dir() {
        FileType::Directory // either type directory or ...
    } else {
        match entry.path().extension() {
            Some(ext) => FileType::File(ext.to_string_lossy().into_owned()), // ... type an actual file (-> extension as String) or ...
            None => FileType::None, // ... no file extension at all
        }
    };

    // size of the file in KB, of folder: 0
    let size: u64 = metadata.len() / 1024;

    // get the last modified time of the file
    let modified_time = match metadata.modified() {
        Ok(time) => time,
        Err(e) => {
            warn!(
                "Failed to get modified time for '{}': {}",
                path.display(),
                e
            );
            // if it's unable to read the modified time, it returns all information currently known
            return FileData {
                name: file_name,
                path,
                last_modified: Local::now(), // last_modified is set to the current time
                file_type,
                size_in_kb: size,
            };
        }
    }; // Convert the last modified time to a readable format
       // Convert SystemTime to DateTime<Local>
    let last_modified = modified_time
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| {
            Local
                .timestamp_opt(d.as_secs() as i64, d.subsec_nanos())
                .single()
                .unwrap_or_else(|| {
                    warn!("Failed to convert timestamp for '{}'", path.display());
                    Local::now()
                })
        })
        .unwrap_or_else(|e| {
            warn!(
                "Failed to get duration since UNIX_EPOCH for '{}': {}",
                path.display(),
                e
            );
            Local::now()
        }); // Fallback to current time if there's an error

    // append the important information to the Vector with the FileEntries
    FileData {
        name: file_name,
        path,
        last_modified,
        file_type,
        size_in_kb: size,
    }
}

#[command]
pub fn format_file_data(path: String) -> Result<Vec<FileDataFormatted>, String> {
    // gets the files from the current path
    let files: Result<Vec<FileData>, String> = list_files_and_folders(path);

    // iterate through every file and format it, so js can work with it
    match files {
        Ok(files) => {
            let mut formatted_files: Vec<FileDataFormatted> = Vec::new();
            for file in files {
                formatted_files.push(file.format());
            }
            Ok(formatted_files)
        }
        Err(error) => Err(error),
    }
}
