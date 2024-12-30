use clipboard::{ClipboardContext, ClipboardProvider};
use std::path::{Path, PathBuf};
use std::fs::{rename, remove_file};
use tauri::command;

#[command]
pub fn copy_file(filepath: String) -> Result<String, String> {
    let path: PathBuf = clean_path(filepath);
    // TODO
    Ok("Copied successfully!".to_string())
}

#[command]
pub fn cut_file(filepath: String) -> Result<String, String> {
    let _: Result<String, String> = match copy_file(filepath.to_owned()) {
        Ok(message) => Ok(message),
        Err(error) => return Err(error.to_string())
    };
    match delete_file(filepath) {
        Ok(_) => Ok("Cut successfully!".to_string()),
        Err(error) => Err(error)
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
        return Err(format!("A file with the name '{}' already exists in the directory.", new_filename));
    }

    // Rename the file
    rename(&path, &new_filepath)
        .map_err(|e| format!("Failed to rename file: {}", e))?;

    Ok("Renamed successfully!".to_string())
}

#[command]
pub fn delete_file(filepath: String) -> Result<String, String> {
    let path: PathBuf = clean_path(filepath);
    // fs::remove_file(filepath).map_err(|e| e.to_string())?;
    Ok("File deleted successfully.".to_string())
}

#[command]
pub fn open_file_with(filepath: String) -> Result<String, String> {
    let path: PathBuf = clean_path(filepath);
    // TODO
    Ok("Copied successfully!".to_string())
}

#[command]
pub fn copy_path(filepath: String) -> Result<String, String> {
    let path: String = normalize_slashes(&filepath.replace("\\", "/")); // ignore the part, where the new PathBuf is created, since not needed
    let mut clipboard: ClipboardContext = ClipboardProvider::new().map_err(|e| e.to_string())?;
    clipboard.set_contents(path.to_string()).map_err(|e| e.to_string())?;
    Ok("Copied successfully!".to_string())
}

/// Bereinigt einen eingegebenen Dateipfad (String) für konsistentes und fehlerfreies Arbeiten.
///
/// - Ersetzt Backslashes (`\`) mit Slashes (`/`) zur Vereinheitlichung.
/// - Entfernt aufeinanderfolgende Slashes (`///` → `/`).
/// - Gibt am Ende einen `PathBuf` zurück.
fn clean_path(filepath: String) -> PathBuf {
    // 1. Ersetze Backslashes durch Forward Slashes (plattformsicher)
    let normalized = filepath.replace("\\", "/");

    // 2. Beseitige aufeinanderfolgende Slashes
    let recomposed = normalize_slashes(&normalized);

    // 3. Konvertiere zu einem sauberen `PathBuf`
    let path = Path::new(&recomposed);

    // Falls der Pfad leer ist oder nur Slash enthält
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

    for ch in path.chars() {
        if ch != '/' || prev_char != '/' {
            result.push(ch);
        }
        prev_char = ch;
    }

    let length = result.len();
    if result[length -1..length] == "/".to_string() {
        result.remove(length);
    }

    result
}