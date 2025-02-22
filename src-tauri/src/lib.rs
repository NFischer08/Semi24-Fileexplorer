// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
pub mod context_actions;
pub mod file_information;
pub mod database_operations;
pub mod manager;
pub mod global_stuff;

use manager::manager_basic_search;
use context_actions::{copy_file, cut_file, delete_file, open_file_with, paste, rename_file, open_file};
use file_information::{ get_file_information, format_file_data };
use database_operations::{check_database, create_database, initialize_database_and_extensions, search_database};
use chrono::{DateTime, Local};
use global_stuff::get_fav_extensions;


#[cfg_attr(feature = "mobile", tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![format_file_data, copy_file, paste, cut_file, delete_file, rename_file, open_file_with, manager_basic_search, get_fav_extensions])
        .run(tauri::generate_context!("tauri.conf.json"))
        .expect("error while running tauri application");
}