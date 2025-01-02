// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/

mod show_files;
mod context_actions;

use show_files::format_file_data;
use context_actions::{copy_file, cut_file, delete_file, rename_file, open_file_with, paste_from_clipboard, copy_to_clipboard};
use chrono::{DateTime, Local};

#[derive(Debug, serde::Serialize)]
enum FileType {
    Directory,
    File(String),
    None,
}

#[derive(Debug)]
struct FileEntry {
    name: String,
    last_modified: DateTime<Local>,
    file_type: FileType,
    size_in_kb: u64
}

#[derive(Debug, serde::Serialize)]
struct FileDataFormatted {
    name: String,
    last_modified: String,
    file_type: String,
    size: String
}

#[cfg_attr(feature = "mobile", tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![format_file_data, copy_file, paste_from_clipboard, copy_to_clipboard, cut_file, delete_file, rename_file, open_file_with])
        .run(tauri::generate_context!("tauri.conf.json"))
        .expect("error while running tauri application");
}
