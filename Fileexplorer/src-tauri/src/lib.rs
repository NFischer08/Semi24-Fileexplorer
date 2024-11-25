// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/

use std::fs;
use std::io;
use std::path::Path;

fn list_files_in_directory(path: &str) -> io::Result<Vec<String>> {
    let mut files = Vec::new();

    // Überprüfen, ob der Pfad existiert und ein Verzeichnis ist
    if Path::new(path).is_dir() {
        // Alle Einträge im Verzeichnis auflisten
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let file_name = entry.file_name();
            files.push(file_name.to_string_lossy().into_owned());
        }
    } else {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Der angegebene Pfad ist kein Verzeichnis oder existiert nicht.",
        ));
    }

    Ok(files)
}

#[tauri::command]
fn greet(name: &str) -> String {
    //format!("Hello, {}! You've been greeted from Rust!", name)
    match list_files_in_directory(name) {
        Ok(files) => {
            // Hier kannst du die Dateien weiterverarbeiten oder zurückgeben
            let files_string = files.join("\n");
            println!("{}", files_string);
            return files_string;
        }
        Err(e) => {
            return format!("Fehler: {}", e);
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
