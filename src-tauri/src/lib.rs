// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
mod manager;
mod file_information;
mod context_actions;

use manager::{manager_create_database, manager_basic_search, manager_check_database};
use file_information::format_file_data;
use context_actions::{cut_file, delete_file, rename_file, open_file_with, paste, copy_file};
use chrono::{DateTime, Local};


#[cfg_attr(feature = "mobile", tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![format_file_data, copy_file, paste, cut_file, delete_file, rename_file, open_file_with])
        .run(tauri::generate_context!("tauri.conf.json"))
        .expect("error while running tauri application");
}
