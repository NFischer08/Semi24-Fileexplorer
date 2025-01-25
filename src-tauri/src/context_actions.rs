use clipboard::{ClipboardContext, ClipboardProvider};
use std::path::{Path, PathBuf};

fn copy(filepath: String) -> Result<String, String> {
    let path: PathBuf = clean_path(filepath);
    Ok("Copied successfully!".to_string())
}

fn cut(filepath: String) -> Result<String, String> {
    let path: PathBuf = clean_path(filepath);
    Ok("Cut successfully!".to_string())
    // delete(path)
}

fn rename(filepath: String) -> Result<String, String> {
    let path: PathBuf = clean_path(filepath);
    Ok("Renamed successfully!".to_string())
}

fn delete(filepath: String) -> Result<String, String> {
    let path: PathBuf = clean_path(filepath);
    // fs::remove_file(filepath).map_err(|e| e.to_string())?;
    Ok("File deleted successfully.".to_owned())
}

fn open_with(filepath: String) -> Result<String, String> {
    let path: PathBuf = clean_path(filepath);
    Ok("Copied successfully!".to_string())
}

fn copy_path(filepath: String) -> Result<String, String> {
    let path: PathBuf = clean_path(filepath);
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

    result
}