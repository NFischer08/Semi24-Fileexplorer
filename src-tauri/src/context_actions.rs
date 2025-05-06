use crate::config_handler::{get_copy_mode, CopyMode};
use crate::manager::CURRENT_DIR;
use clipboard::{ClipboardContext, ClipboardProvider};
use copy_dir::copy_dir;
use opener::open;
use std::{
    fs,
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use tauri::command;
use crate::rt_db_update::delete_from_db;

#[command]
pub fn copy_file(filepath: String) -> Result<String, String> {
    // get needed values
    let source_path: PathBuf = clean_path(filepath);
    let mut copy_mode: CopyMode = get_copy_mode();
    let mut copy_path = CURRENT_DIR.clone();
    copy_path.push("data/tmp/");

    // check if the source path is valid
    if !source_path.exists() {
        return Err(String::from("No such file or directory"));
    }

    // if a directory gets copyed, it must be done `File` mode
    if source_path.is_dir() {
        copy_mode = CopyMode::File
    }

    match copy_mode {
        CopyMode::Clipboard => {
            // attempt to open the file
            let mut file = match fs::File::open(&source_path) {
                Ok(file) => file,
                Err(_) => {
                    return Err("Failed to open file.".to_string());
                }
            };

            // read the file contents
            let mut contents = String::new();
            match file.read_to_string(&mut contents) {
                Ok(_) => {}
                Err(_) => {
                    return Err("Failed to read file.".to_string());
                }
            };

            // create a clipboard context
            let mut clipboard: ClipboardContext = match ClipboardProvider::new() {
                Ok(ctx) => ctx,
                Err(_) => {
                    return Err("Failed to access clipboard.".to_string());
                }
            };

            // copy the contents to the clipboard
            match clipboard.set_contents(contents) {
                Ok(_) => {}
                Err(_) => {
                    return Err("Failed to copy to clipboard.".to_string());
                }
            };
        }
        CopyMode::File => {
            // remove previous file or folder
            if copy_path.join("CONTENT").exists() {
                delete_file(copy_path.join("CONTENT").to_string_lossy().to_string())?;
            }
            // copy file or folder to paste it later
            match copy_dir(&source_path, &copy_path.join("CONTENT")) {
                Ok(_) => {}
                Err(e) => return Err(e.to_string()),
            };
        }
    }
    // get filename
    let filename: String = match source_path.file_name() {
        Some(name) => name.to_string_lossy().to_string(),
        None => String::from("Unbenannt"),
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
        Err(e) => return Err(e.to_string()),
    };

    // write the content
    match file.write(filename.as_ref()) {
        Ok(_) => {}
        Err(e) => return Err(e.to_string()),
    }

    Ok(format!(
        "File copyied successfully to {}!",
        source_path.display()
    ))
}

#[command]
pub fn paste_file(destination: String) -> Result<String, String> {
    // get needed values
    let mut dest_path: PathBuf = clean_path(destination);
    let copy_mode: CopyMode = get_copy_mode();
    let mut copy_path = CURRENT_DIR.clone();
    copy_path.push("data/tmp/");

    // incase it's an file, the parent dir is needed
    if dest_path.is_file() {
        dest_path = match dest_path.parent() {
            None => return Err(String::from("Failed to get parent directory.")),
            Some(parent) => parent.to_path_buf(),
        };
    }

    // read filename
    let file = match fs::File::open(copy_path.join("copy.txt")) {
        Ok(file) => file,
        Err(e) => return Err(e.to_string()),
    };
    let mut reader = BufReader::new(file);
    let mut filename = String::new();
    match reader.read_to_string(&mut filename) {
        Ok(_) => {}
        Err(_) => filename = String::from("Unbenannt"),
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
                Err(_) => return Err("Failed to access clipboard.".to_string()),
            };

            // get the contents from the clipboard
            let contents = match clipboard.get_contents() {
                Ok(contents) => contents,
                Err(_) => return Err("Failed to read clipboard.".to_string()),
            };

            // create the file
            let mut file = match fs::File::create(&dest_path) {
                Ok(file) => file,
                Err(e) => return Err(e.to_string()),
            };

            // write contents to the file
            match file.write_all(contents.as_bytes()) {
                Ok(_) => {}
                Err(_) => return Err("Failed write to file!".to_string()),
            }
        }
        CopyMode::File => {
            // copy file to desired location
            match copy_dir(copy_path.join("CONTENT"), &dest_path) {
                Ok(_) => {}
                Err(e) => return Err(e.to_string()),
            };
        }
    }

    Ok(String::from("Successfully copied file!"))
}

#[command]
pub fn cut_file(filepath: String) -> Result<String, String> {
    match copy_file(filepath.to_owned()) {
        Ok(_) => {}
        Err(error) => return Err(error), // returnt Error, wenn kopieren nicht klappt -> nicht wird gelöscht!
    };
    match delete_file(filepath) {
        //nicht lieber erstmal überprüfen, ob die auch wieder gespeichert wurde? Nino: wenns nicht klappt wird ein Error returnt!
        Ok(_) => Ok("Cut successfully!".to_string()),
        Err(error) => Err(error),
    }
}

#[command]
pub fn rename_file(filepath: String, new_filename: &str) -> Result<String, String> {
    // Clean the path
    let path: PathBuf = clean_path(filepath);

    // Get the parent directory
    let parent = path.parent().ok_or("Failed to get parent directory")?;

    // Create the new file path without a leading slash
    let new_filepath: PathBuf = parent.join(new_filename);

    // Check if the new file already exists
    if new_filepath.exists() {
        return Err(format!(
            "A file with the name '{}' already exists in the directory.",
            new_filename
        ));
    }

    // Rename the file
    fs::rename(&path, &new_filepath).map_err(|e| format!("Failed to rename file: {}", e))?;

    Ok("Renamed successfully!".to_string())
}

#[command]
pub fn delete_file(filepath: String) -> Result<String, String> {
    let path: PathBuf = clean_path(filepath);

    // check if path is valid
    if !path.exists() {
        return Err(String::from("No such file or directory."));
    }

    // delete dir / file
    if path.is_dir() {
        //fs::remove_dir_all(&path).map_err(|e| e.to_string())?;
        // get the connection pool from manager
        let connection_pool: Pool<SqliteConnectionManager> = manager_make_pooled_connection();
        // get a valid connection to db and remove just deleted folder from db
        match connection_pool.get() {
            Ok(conn) => delete_from_db(&conn, &path),
            Err(_) => {},
        }
        println!("Deleted directory '{}'", path.display());
    } else if path.is_file() {
        //fs::remove_file(path).map_err(|e| e.to_string())?;
        println!("Deleted file '{}'", path.display());
    } else {
        return Err(String::from("File doesn't exist."));
    }
    Ok("File deleted successfully.".to_string())
}

#[command]
pub fn open_file(filepath: String) -> Result<String, String> {
    let path: PathBuf = clean_path(filepath);
    match open(path) {
        Ok(_) => Ok(String::from("File opened successfully!")),
        Err(_) => Err(String::from("Failed to open file for user.")),
    }
}

/// Bereinigt einen eingegebenen Dateipfad (String) für konsistentes und fehlerfreies Arbeiten.
///
/// - Ersetzt Backslashes (`\`) mit Slashes (`/`) zur Vereinheitlichung.
/// - Entfernt aufeinanderfolgende Slashes (`//` → `/`).
/// - Gibt am Ende einen `PathBuf` zurück.
fn clean_path(filepath: String) -> PathBuf {
    // 1. Beseitige aufeinanderfolgende Slashes und Backslahes
    let recomposed = normalize_slashes(&filepath);

    // 2. Konvertiere zu einem sauberen `PathBuf`
    let path = Path::new(&recomposed);

    // Falls der Pfad leer ist oder nur Slash enthält -> kann eig. nicht passieren
    if path.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        path.to_path_buf()
    }
}

/// Hilfsfunktion: Entfernt aufeinanderfolgende Slashes.
fn normalize_slashes(path: &str) -> String {
    let mut result = String::new();
    let mut prev_char = '\0';
    let path: &str = &path.replace("\\", "/");

    for ch in path.chars() {
        if ch != '/' || prev_char != '/' {
            result.push(ch);
        }
        prev_char = ch;
    }
    // removing unused / at the end
    //let length = result.len();
    //if result[length -1..length] == "/".to_string() {
    //    result.remove(length);
    //}

    result
}
