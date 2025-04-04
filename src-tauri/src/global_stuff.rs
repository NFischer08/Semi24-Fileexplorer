use crate::config_handler::FAVOURITE_FILE_EXTENSIONS;
use std::collections::HashMap;
use tauri::command;

#[command]
pub fn get_fav_extensions() -> Option<HashMap<String, String>> {
    let x = if let Some(fav_ext) = FAVOURITE_FILE_EXTENSIONS.get() {
        Some(fav_ext.clone())
    } else {
        let mut x = HashMap::new();
        x.insert("dirs".to_string(), "dir".to_string());
        Some(x)

    };
    println!("GLOB: fav_ext: {:?}", x);
    x
}
