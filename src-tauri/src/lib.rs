// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
pub mod context_actions;
pub mod database_operations;
pub mod file_information;
pub mod global_stuff;
pub mod manager;
pub mod rt_db_update;

use context_actions::{copy_file, cut_file, delete_file, open_file_with, paste, rename_file};
use file_information::format_file_data;
use global_stuff::get_fav_extensions;
use manager::manager_basic_search;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            format_file_data,
            copy_file,
            paste,
            cut_file,
            delete_file,
            rename_file,
            open_file_with,
            manager_basic_search,
            get_fav_extensions
        ])
        .run(tauri::generate_context!("tauri.conf.json"))
        .expect("error while running tauri application");
}
