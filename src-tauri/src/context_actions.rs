use regex::Regex;
use clipboard::{ClipboardContext, ClipboardProvider};

fn copy(filepath: String) -> Result<String, String> {
    let path: String = clean_path(filepath);
    Ok("Copied successfully!".to_string())
}

fn cut(filepath: String) -> Result<String, String> {
    let path: String = clean_path(filepath);
    delete(path)
}

fn rename(filepath: String) -> Result<String, String> {
    let path: String = clean_path(filepath);
    Ok("Copied successfully!".to_string())
}

fn delete(filepath: String) -> Result<String, String> {
    let path: String = clean_path(filepath);
    Ok("Copied successfully!".to_string())
}

fn open_with(filepath: String) -> Result<String, String> {
    let path: String = clean_path(filepath);
    Ok("Copied successfully!".to_string())
}

fn copy_path(filepath: String) -> Result<String, String> {
    let path: String = clean_path(filepath);
    let mut clipboard: ClipboardContext = ClipboardProvider::new().map_err(|e| e.to_string())?;
    clipboard.set_contents(path.to_string()).map_err(|e| e.to_string())?;
    Ok("Copied successfully!".to_string())
}

fn clean_path(filepath: String) -> String {
    // Replace backslashes with slashes
    let path = filepath.replace("\\", "/");

    // Create a regex to match multiple consecutive slashes
    let re = Regex::new(r"/+").unwrap();

    // Replace multiple consecutive slashes with a single slash
    let cleaned_path = re.replace_all(&path, "/").to_string();

    // Return the cleaned path
    cleaned_path
}