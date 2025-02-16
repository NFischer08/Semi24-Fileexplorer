// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod manager;

use manager::{manager_create_database, manager_check_database};
use std::thread;
use std::time::Duration;

fn main() {
    let thread_creating_database = thread::spawn(|| {
        thread::sleep(Duration::from_millis(750));
        manager_create_database("/").unwrap();
        manager_check_database().unwrap();
    });

    file_explorer_lib::run();

    thread_creating_database.join().unwrap();
}
