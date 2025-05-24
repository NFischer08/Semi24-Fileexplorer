// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
pub mod config_handler;
pub mod context_actions;
pub mod db_create;
pub mod db_search;
pub mod db_util;
pub mod file_information;
pub mod manager;
pub mod rt_db_update;

fn main() {
    file_explorer_lib::run();
}
