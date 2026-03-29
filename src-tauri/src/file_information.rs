use crate::context_actions::clean_path;
use chrono::{DateTime, Local, TimeZone};
use log::{error, warn};
use std::{
    fs::{self, DirEntry, Metadata},
    path::PathBuf,
    time::SystemTime,
};
use tauri::command;

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
    pub(crate) name: String,
    pub(crate) path: String,
    pub(crate) last_modified: String,
    pub(crate) file_type: String,
    pub(crate) size: String,
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
            format!("{rounded_size:.1} {unit}")
        };
        FileDataFormatted {
            name: self.name,
            path: self.path.to_string_lossy().to_string().replace("\\", "/"),
            last_modified: self.last_modified.format("%d.%m.%Y %H:%M").to_string(),
            file_type,
            size,
        }
    }

    fn construct(path: PathBuf, metadata: Metadata) -> FileData {
        // get the filetype of a file
        let file_type = if metadata.is_dir() {
            FileType::Directory // either type directory or ...
        } else {
            match path.extension() {
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
                    name: path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned(),
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
            name: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            path,
            last_modified,
            file_type,
            size_in_kb: size,
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

impl From<PathBuf> for FileData {
    fn from(path: PathBuf) -> FileData {
        let metadata: Metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(e) => {
                warn!("Failed to get metadata for '{}': {e}", path.display());
                return FileData {
                    name: path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned(),
                    path,
                    ..FileData::default()
                };
            }
        };
        FileData::construct(path, metadata)
    }
}

impl From<&DirEntry> for FileData {
    fn from(entry: &DirEntry) -> FileData {
        let metadata: Metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(e) => {
                warn!(
                    "Failed to get metadata for '{}': {e}",
                    entry.path().display()
                );
                return FileData {
                    name: entry.file_name().into_string().unwrap_or_default(),
                    path: entry.path(),
                    ..FileData::default()
                };
            }
        };
        FileData::construct(entry.path(), metadata)
    }
}

fn list_files_and_folders(filepath: String) -> Result<Vec<FileData>, String> {
    let path = clean_path(filepath);

    // Check if the path exists
    if !path.exists() {
        return Err(String::from("The specified path does not exist."));
    }

    // Check if the path is a directory
    if !path.is_dir() {
        return Err(String::from("The specified path is not a directory."));
    }

    if !path.to_string_lossy().to_string().contains("/") {
        warn!(
            "The specified path '{}' is not valid. Seems like you forgot a slash.",
            path.to_string_lossy()
        );
        return Err(String::from(
            "The specified path is not valid. Seems like you forgot a slash.",
        ));
    }

    let mut entries: Vec<FileData> = Vec::new();

    // Read the directory entries
    match fs::read_dir(&path) {
        Ok(dir_entries) => {
            for entry in dir_entries {
                match entry {
                    Ok(entry) => entries.push(FileData::from(&entry)),
                    Err(e) => {
                        error!("Failed to read directory entry: {e}");
                        return Err(e.to_string());
                    }
                }
            }
        }
        Err(e) => {
            error!("Failed to read directory '{}': {e}", path.display());
            return Err(e.to_string());
        }
    }

    Ok(entries)
}

#[command]
pub fn format_file_data(path: String) -> Result<Vec<FileDataFormatted>, String> {
    // gets the files from the current path
    let files: Vec<FileData> = list_files_and_folders(path)?;

    // iterate through every file and format it, so js can work with it
    Ok(files.into_iter().map(|f| f.format()).collect())
}
