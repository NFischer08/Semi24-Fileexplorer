use clipboard::{ClipboardContext, ClipboardProvider};
use opener::open;
use std::{
    fs::{self, remove_file, rename, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
};
use tauri::command;

#[command]
pub fn copy_file(filepath: String) -> Result<String, String> {
    let clean_path: PathBuf = clean_path(filepath);
    let mode: u8 = 2;
    match mode {
        1 => copy_from_file(clean_path),
        2 => copy_from_path(clean_path),
        _ => Err(String::from("Invalid mode!")),
    }
}

fn copy_from_file(path: PathBuf) -> Result<String, String> {
    // checks if path refers to a directory
    if path.is_dir() {
        return Err(String::from("Can't copy directory! yet!"));
    }

    println!("copying file {}", path.display());
    // Attempt to open the file
    let mut file = match File::open(&path) {
        Ok(file) => file,
        Err(_) => {
            return Err("Failed to open file.".to_string());
        }
    };

    // Read the file contents
    let mut contents = String::new();
    match file.read_to_string(&mut contents) {
        Ok(_) => {}
        Err(_) => {
            return Err("Failed to read file.".to_string());
        }
    };

    // Append filename and fileextension to the content
    let filename: String = match path.file_name() {
        Some(name) => format!("={}", name.to_string_lossy()),
        None => String::from("=Unbenannt"),
    };
    println!("Filename: {}", filename);

    contents += filename.as_str();

    // Create a clipboard context
    let mut clipboard: ClipboardContext = match ClipboardProvider::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            return Err("Failed to access clipboard.".to_string());
        }
    };

    // Copy the contents to the clipboard
    match clipboard.set_contents(contents) {
        Ok(_) => {}
        Err(_) => {
            return Err("Failed to copy to clipboard.".to_string());
        }
    };

    println!("Copied successfully!");
    Ok(format!("File copyied successfully to {}!", path.display()))
}

fn copy_from_path(path: PathBuf) -> Result<String, String> {
    // Copying path to Clipboard
    if !path.exists() {
        // Pfad kann nur existieren, da er sonnst nicht übergeben werden kann! kann also eigentlich weg
        return Err("Source file does not exist.".to_string());
    }

    let mut clipboard: ClipboardContext =
        ClipboardProvider::new().map_err(|e| format!("Failed to access clipboard: {}", e))?;

    clipboard
        .set_contents(path.to_string_lossy().into_owned()) // geht alles einfacher, indem man den Pfad nie zum PathBuf macht
        .map_err(|e| format!("Failed to copy to clipboard: {}", e))?;

    Ok("File path copied to clipboard.".to_string())
}

#[command]
pub fn paste(destination: String) -> Result<String, String> {
    let dest_path: PathBuf = clean_path(destination);
    let mode: u8 = 2;
    match mode {
        1 => paste_from_file(dest_path),
        2 => paste_from_path(dest_path),
        _ => Err("Invalid mode!".to_string()),
    }
}

fn paste_from_file(destination: PathBuf) -> Result<String, String> {
    // File already exists? TODO => replace, keep both, cancel
    // => like Linux? => user has to enter other filename
    //if destination.exists() { => only folder not file => always throws an error
    //    println!("File already exists!; {}", destination.display());
    //    return Err(String::from("File already exists!"));
    //}
    // Create a clipboard context
    println!(
        "Pasting: {} ... next: Access clipboard",
        destination.display()
    );
    let mut clipboard: ClipboardContext = match ClipboardProvider::new() {
        Ok(ctx) => ctx,
        Err(_) => return Err("Failed to access clipboard.".to_string()),
    };
    println!("Accessed clipboard... next: read contents");

    // Get the contents from the clipboard
    let contents = match clipboard.get_contents() {
        Ok(contents) => contents,
        Err(_) => return Err("Failed to read clipboard.".to_string()),
    };
    println!("Read Contents \"{}\" ... next: create File", contents);

    let (name, index) = match contents.rfind('=') {
        Some(index) => (format!("{}", &contents[index + 1..]), index),
        None => (String::from("Unbenannt"), contents.len()),
    };
    println!("Name: {}", name);
    let contents: String = contents[..index].to_string();

    // Write the contents to the specified file
    let mut file = match File::create(&destination.join(name)) {
        Ok(file) => file,
        Err(e) => return Err(e.to_string()),
    };
    println!("Created File ... next: write contents");

    match file.write_all(contents.as_bytes()) {
        Ok(_) => Ok(format!(
            "Successfully copied file to {}",
            destination.display()
        )),
        Err(_) => Err("Failed write to file!".to_string()),
    }
}

fn paste_from_path(dest_path: PathBuf) -> Result<String, String> {
    let mut clipboard: ClipboardContext =
        ClipboardProvider::new().map_err(|e| format!("Failed to access clipboard: {}", e))?;

    // Dateipfad aus der Zwischenablage lesen
    let source_path_str = clipboard
        .get_contents()
        .map_err(|e| format!("Failed to read from clipboard: {}", e))?;

    let source_path = Path::new(&source_path_str); // pastet aktuell in den Zielpath -> nicht aktueller!

    // Überprüfen, ob die Quelldatei existiert
    if !source_path.exists() {
        return Err("Source file does not exist.".to_string());
    }

    // Sicherstellen, dass das Zielverzeichnis existiert
    if let Some(parent) = dest_path.parent() {
        if !parent.exists() {
            match fs::create_dir_all(parent) {
                Ok(_) => {}
                Err(e) => return Err(format!("Failed to create destination directory: {}", e)),
            };
        }
    }

    // Datei kopieren
    match fs::copy(&source_path, &dest_path) {
        Ok(_) => println!("Successfully copied file to {}", dest_path.display()),
        Err(_) => return Err("Failed to copy file!".to_string()),
    }

    Ok(format!(
        "File pasted successfully to {}.",
        dest_path.display()
    ))
}

#[command]
pub fn cut_file(filepath: String) -> Result<String, String> {
    match copy_from_file(clean_path(filepath.to_owned())) {
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
    rename(&path, &new_filepath).map_err(|e| format!("Failed to rename file: {}", e))?;

    Ok("Renamed successfully!".to_string())
}

#[command]
pub fn delete_file(filepath: String) -> Result<String, String> {
    let _path: PathBuf = clean_path(filepath);
    //remove_file(path).map_err(|e| e.to_string())?;
    Ok("File deleted successfully.".to_string())
}

#[command]
pub fn open_file_with(filepath: String) -> Result<String, String> {
    open_file_with_complicated(filepath)
}

fn open_file_with_complicated(filepath: String) -> Result<String, String> {
    let path: PathBuf = clean_path(filepath);
    #[cfg(target_os = "windows")]
    {
        match Command::new("cmd")
            .args(&["/C", "start", "", path.to_str().unwrap()])
            .spawn()
        {
            Ok(_) => Ok("File opened successfully.".to_string()),
            Err(_) => Err("Failed to open file.".to_string()),
        }
    }

    #[cfg(target_os = "macos")]
    {
        match Command::new("open").arg(path).spawn() {
            Ok(_) => Ok("File opened successfully.".to_string()),
            Err(_) => Err("Failed to open file.".to_string()),
        }
    }

    #[cfg(target_os = "linux")]
    {
        match Command::new("xdg-open").arg(path).spawn() {
            Ok(_) => Ok("File opened successfully.".to_string()),
            Err(_) => Err("Failed to open file.".to_string()),
        }
    }
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
    //let length = result.len();
    //if result[length -1..length] == "/".to_string() {
    //    result.remove(length);
    //}

    result
}
