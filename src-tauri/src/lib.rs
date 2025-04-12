// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
pub mod config_handler;
pub mod context_actions;
pub mod db_create;
pub mod db_search;
pub mod db_util;
pub mod file_information;
pub mod manager;

use tauri::Manager;
use config_handler::{get_css_settings, get_fav_file_extensions};
use context_actions::{
    copy_file, cut_file, delete_file, open_file, open_file_with, paste_file, rename_file,
};
use file_information::format_file_data;
use manager::manager_basic_search;
use crate::manager::AppState;

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            app.manage(AppState {
                handle: app.handle().clone()
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            format_file_data,
            copy_file,
            paste_file,
            cut_file,
            delete_file,
            rename_file,
            open_file,
            open_file_with,
            manager_basic_search,
            get_fav_file_extensions,
            get_css_settings
        ])
        .run(tauri::generate_context!("tauri.conf.json"))
        .expect("error while running tauri application");
}
