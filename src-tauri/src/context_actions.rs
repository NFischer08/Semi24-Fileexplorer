use crate::config_handler::{get_copy_mode, CopyMode, CURRENT_DIR};
use crate::db_util::delete_from_db;
use crate::manager::manager_make_connection_pool;
use clipboard::{ClipboardContext, ClipboardProvider};
use copy_dir::copy_dir;
use log::{error, info, warn};
use opener::open;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use std::{
    fs,
    io::{BufReader, Read, Write},
    path::PathBuf,
};
use tauri::command; // <-- Add log macros

/// copes a file either to the clipboard or as a file in the tmp folder
#[command]
pub fn copy_file(filepath: String) -> Result<(), String> {
    // get necessary values
    let source_path: PathBuf = clean_path(filepath);
    let mut copy_mode: CopyMode = get_copy_mode();
    let mut copy_path = CURRENT_DIR.clone();
    copy_path.push("data/tmp/");

    // check if the source path is valid
    if !source_path.exists() {
        warn!("Couldn't find file or directory: {}", source_path.display());
        return Err(String::from("Couldn't find file or directory."));
    }

    // if a directory gets copied, it must be done `File` mode
    if source_path.is_dir() {
        copy_mode = CopyMode::File
    }

    match copy_mode {
        CopyMode::Clipboard => {
            // attempt to open the file
            let mut file = match fs::File::open(&source_path) {
                Ok(file) => file,
                Err(e) => {
                    error!("Failed to open file '{}': {}", source_path.display(), e);
                    return Err("Failed to open file.".to_string());
                }
            };

            // read the file contents
            let mut contents = String::new();
            match file.read_to_string(&mut contents) {
                Ok(_) => {}
                Err(e) => {
                    error!("Failed to read file '{}': {}", source_path.display(), e);
                    return Err("Failed to read file.".to_string());
                }
            };

            // create a clipboard context
            let mut clipboard: ClipboardContext = match ClipboardProvider::new() {
                Ok(ctx) => ctx,
                Err(e) => {
                    error!("Failed to access clipboard: {}", e);
                    return Err("Failed to access clipboard.".to_string());
                }
            };

            // copy the contents to the clipboard
            match clipboard.set_contents(contents) {
                Ok(_) => {
                    info!(
                        "Copied contents of '{}' to clipboard.",
                        source_path.display()
                    );
                }
                Err(e) => {
                    error!("Failed to copy to clipboard: {}", e);
                    return Err("Failed to copy to clipboard.".to_string());
                }
            };
        }
        CopyMode::File => {
            // remove previous file or folder
            if copy_path.join("CONTENT").exists() {
                if let Err(e) = delete_file(copy_path.join("CONTENT").to_string_lossy().to_string())
                {
                    error!("Failed to delete previous CONTENT: {}", e);
                    return Err(e);
                }
            }
            // copy a file or folder to paste it later
            if let Err(e) = copy_dir(&source_path, copy_path.join("CONTENT")) {
                error!("Failed to copy dir/file: {}", e);
                return Err(e.to_string());
            }
        }
    }
    // get filename
    let filename: String = match source_path.file_name() {
        Some(name) => name.to_string_lossy().to_string(),
        None => {
            warn!("File has no name, using 'Unnamed'");
            String::from("Unnamed")
        }
    };

    // open copy file to store filename
    let mut file = match fs::OpenOptions::new()
        .read(false)
        .truncate(true)
        .write(true)
        .create(true)
        .open(copy_path.join("copy.txt"))
    {
        Ok(file) => file,
        Err(e) => {
            error!("Failed to open copy.txt for writing: {}", e);
            return Err(e.to_string());
        }
    };

    // write the content
    match file.write(filename.as_ref()) {
        Ok(_) => {}
        Err(e) => {
            error!("Failed to write filename to copy.txt: {}", e);
            return Err(e.to_string());
        }
    }

    Ok(())
}

/// pastes a file either to the clipboard or as a file in the tmp folder
#[command]
pub fn paste_file(destination: String) -> Result<(), String> {
    // get necessary values
    let mut dest_path: PathBuf = clean_path(destination);
    let copy_mode: CopyMode = get_copy_mode();
    let mut copy_path = CURRENT_DIR.clone();
    copy_path.push("data/tmp/");

    // incase it's a file, the parent dir is needed
    if dest_path.is_file() {
        dest_path = match dest_path.parent() {
            None => {
                error!(
                    "Failed to get parent directory for '{}'",
                    dest_path.display()
                );
                return Err(String::from("Failed to get parent directory."));
            }
            Some(parent) => parent.to_path_buf(),
        };
    }

    // read filename
    let file = match fs::File::open(copy_path.join("copy.txt")) {
        Ok(file) => file,
        Err(e) => {
            error!("Failed to open copy.txt: {}", e);
            return Err(e.to_string());
        }
    };
    let mut reader = BufReader::new(file);
    let mut filename = String::new();
    match reader.read_to_string(&mut filename) {
        Ok(_) => {}
        Err(e) => {
            warn!(
                "Failed to read filename from copy.txt: {}, using 'Unnamed'",
                e
            );
            filename = String::from("Unnamed");
        }
    };

    // make sure the filename doesn't exist yet
    dest_path = dest_path.join(&filename);
    let mut counter: u32 = 1;
    while dest_path.exists() {
        dest_path.set_file_name(format!("new {counter} {filename}"));
        counter += 1;
    }

    match copy_mode {
        CopyMode::Clipboard => {
            // open clipboard
            let mut clipboard: ClipboardContext = match ClipboardProvider::new() {
                Ok(ctx) => ctx,
                Err(e) => {
                    error!("Failed to access clipboard: {}", e);
                    return Err("Failed to access clipboard.".to_string());
                }
            };

            // get the contents from the clipboard
            let contents = match clipboard.get_contents() {
                Ok(contents) => contents,
                Err(e) => {
                    error!("Failed to read clipboard: {}", e);
                    return Err("Failed to read clipboard.".to_string());
                }
            };

            // create the file
            let mut file = match fs::File::create(&dest_path) {
                Ok(file) => file,
                Err(e) => {
                    error!("Failed to create file '{}': {}", dest_path.display(), e);
                    return Err(e.to_string());
                }
            };

            // write contents to the file
            match file.write_all(contents.as_bytes()) {
                Ok(_) => {}
                Err(e) => {
                    error!("Failed to write to file '{}': {}", dest_path.display(), e);
                    return Err("Failed write to file!".to_string());
                }
            }
        }
        CopyMode::File => {
            // copy file to desired location
            match copy_dir(copy_path.join("CONTENT"), &dest_path) {
                Ok(_) => {}
                Err(e) => {
                    error!(
                        "Failed to copy dir/file to '{}': {}",
                        dest_path.display(),
                        e
                    );
                    return Err(e.to_string());
                }
            };
        }
    }

    Ok(())
}

/// cuts a file by combining copy and delete
#[command]
pub fn cut_file(filepath: String) -> Result<(), String> {
    copy_file(filepath.to_owned())?;
    delete_file(filepath)
}

/// renames an old file path to the new one
#[command]
pub fn rename_file(filepath: String, new_filename: &str) -> Result<(), String> {
    // Clean the path
    let path: PathBuf = clean_path(filepath);

    // Get the parent directory
    let parent = path.parent().ok_or_else(|| {
        error!("Failed to get parent directory for '{}'", path.display());
        "Failed to get parent directory"
    })?;

    // Create the new file path without a leading slash
    let new_filepath: PathBuf = parent.join(new_filename);

    // Check if the new file already exists
    if new_filepath.exists() {
        warn!(
            "A file with the name '{}' already exists in the directory.",
            new_filename
        );
        return Err(format!(
            "A file with the name '{}' already exists in the directory.",
            new_filename
        ));
    }

    // Rename the file
    fs::rename(&path, &new_filepath).map_err(|e| {
        error!("Failed to rename file: {}", e);
        e.to_string()
    })?;

    Ok(())
}

/// deletes the given file path from fs and db
#[command]
pub fn delete_file(filepath: String) -> Result<(), String> {
    let path: PathBuf = clean_path(filepath);

    // check if a path is valid
    if !path.exists() {
        warn!("Couldn't find file or directory: {}", path.display());
        return Err(String::from("Couldn't find file or directory."));
    }

    // delete dir / file
    if path.is_dir() {
        fs::remove_dir_all(&path).map_err(|e| {
            error!("Failed to remove directory '{}': {}", path.display(), e);
            e.to_string()
        })?;
        // get the connection pool from the manager
        let connection_pool: Pool<SqliteConnectionManager> = manager_make_connection_pool();
        // get a valid connection to db and remove just deleted folder from db
        if let Ok(conn) = connection_pool.get() {
            delete_from_db(&conn, &path) //
        }
    } else if path.is_file() {
        fs::remove_file(&path).map_err(|e| {
            error!("Failed to remove file '{}': {}", path.display(), e);
            e.to_string()
        })?;
    } else {
        error!("Problem when detecting type for '{}'", path.display());
        return Err(String::from("Problem when detecting type."));
    }
    Ok(())
}

/// opens the file for the user
#[command]
pub fn open_file(filepath: String) -> Result<(), String> {
    let path: PathBuf = clean_path(filepath);
    match open(path) {
        Ok(_) => Ok(()),
        Err(e) => {
            error!("Failed to open file: {}", e);
            Err(e.to_string())
        }
    }
}

/// cleans and formats a given path (`String`) to a `PathBuf`
pub fn clean_path(filepath: String) -> PathBuf {
    // remove double slashes and backslashes
    let filepath: &str = &filepath.replace("\\", "/");
    let parts: Vec<&str> = filepath
        .split('/')
        .filter(|&s| !s.is_empty() && s != ".")
        .collect();

    // on windows the path has to start with C:
    #[cfg(target_os = "windows")]
    {
        return if filepath.starts_with("/") {
            PathBuf::from("C:/").join(parts.join("/"))
        } else {
            let path = PathBuf::from(parts.join("/"));
            if !path.to_string_lossy().contains("/") {
                return PathBuf::from(format!("{}/", path.to_string_lossy()));
            }
            return path;
        };
    }
    PathBuf::from("/").join(parts.join("/"))
}
