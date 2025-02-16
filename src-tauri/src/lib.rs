// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
mod manager;
mod context_actions;

use manager::{ file_information::{get_file_information, format_file_data}, manager_basic_search };
use context_actions::{copy_file, cut_file, delete_file, open_file_with, paste, rename_file};
use chrono::{DateTime, Local};


#[cfg_attr(feature = "mobile", tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![format_file_data, copy_file, paste, cut_file, delete_file, rename_file, open_file_with, manager_basic_search])
        .run(tauri::generate_context!("tauri.conf.json"))
        .expect("error while running tauri application");
}