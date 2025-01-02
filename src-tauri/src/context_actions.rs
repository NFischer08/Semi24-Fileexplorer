use clipboard::{ClipboardContext, ClipboardProvider};
use std::path::{Path, PathBuf};
use std::fs::{self, File, rename, remove_file};
use tauri::command;
use std::io::Write;

#[command]
pub fn copy_file(filepath: String) -> Result<String, String> {
    let _path: PathBuf = clean_path(filepath);
    // TODO
    Ok("Copied successfully!".to_string())
}

#[command]
pub fn copy_path(filepath: String) -> Result<String, String> { // Copying path to Clipboard
    let path: PathBuf = clean_path(filepath); // Pfad bereinigen

    if !path.exists() { // Pfad kann nur existieren, da er sonnst nicht übergeben werden kann! kann also eigentlich weg
        return Err("Source file does not exist.".to_string());
    }

    let mut clipboard: ClipboardContext = ClipboardProvider::new()
        .map_err(|e| format!("Failed to access clipboard: {}", e))?;

    clipboard
        .set_contents(path.to_string_lossy().into_owned()) // geht alles einfacher, indem man den Pfad nie zum PathBuf macht
        .map_err(|e| format!("Failed to copy to clipboard: {}", e))?;

    Ok("File path copied to clipboard.".to_string())
}

#[command]
pub fn paste_from_path(destination: String) -> Result<String, String> {
    let mut clipboard: ClipboardContext = ClipboardProvider::new()
        .map_err(|e| format!("Failed to access clipboard: {}", e))?;

    // Dateipfad aus der Zwischenablage lesen
    let source_path_str = clipboard
        .get_contents()
        .map_err(|e| format!("Failed to read from clipboard: {}", e))?;

    let source_path = Path::new(&source_path_str);
    let dest_path: PathBuf = clean_path(destination); // pastet aktuell in den Zielpath -> nicht aktueller!

    // Überprüfen, ob die Quelldatei existiert
    if !source_path.exists() {
        return Err("Source file does not exist.".to_string());
    }

    // Sicherstellen, dass das Zielverzeichnis existiert
    if let Some(parent) = dest_path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| {
                format!("Failed to create destination directory: {}", e)
            })?;
        }
    }

    // Datei kopieren
    fs::copy(&source_path, &dest_path)
        .map_err(|e| format!("Failed to paste file: {}", e))?;

    Ok(format!(
        "File pasted successfully to {}.",
        dest_path.display()
    ))
}

#[command]
pub fn cut_file(filepath: String) -> Result<String, String> {
    let _: Result<String, String> = match copy_file(filepath.to_owned()) {
        Ok(message) => Ok(message),
        Err(error) => return Err(error.to_string()) // returnt Error, wenn kopieren nicht klappt -> nicht wird gelöscht!
    };
    match delete_file(filepath) { //nicht lieber erstmal überprüfen, ob die auch wieder gespeichert wurde? Nino: wenns nicht klappt wird ein Error returnt!
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
    let _path: PathBuf = clean_path(filepath);
    // fs::remove_file(filepath).map_err(|e| e.to_string())?;
    Ok("File deleted successfully.".to_string())
}

#[command]
pub fn open_file_with(filepath: String) -> Result<String, String> {
    let _path: PathBuf = clean_path(filepath);
    // TODO
    Ok("Copied successfully!".to_string())
}
/*
#[command]
pub fn copy_path_old(filepath: String) -> Result<String, String> {
    let path: String = normalize_slashes(&filepath.replace("\\", "/")); // ignore the part, where the new PathBuf is created, since not needed
    let mut clipboard: ClipboardContext = ClipboardProvider::new().map_err(|e| e.to_string())?;
    clipboard.set_contents(path.to_string()).map_err(|e| e.to_string())?;
    Ok("Copied successfully!".to_string())
} */

/// Bereinigt einen eingegebenen Dateipfad (String) für konsistentes und fehlerfreies Arbeiten.
///
/// - Ersetzt Backslashes (`\`) mit Slashes (`/`) zur Vereinheitlichung.
/// - Entfernt aufeinanderfolgende Slashes (`///` → `/`).
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
    let length = result.len();
    if result[length -1..length] == "/".to_string() {
        result.remove(length);
    }

    result
}