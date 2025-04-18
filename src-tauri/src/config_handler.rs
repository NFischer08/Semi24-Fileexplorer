use crate::manager::CURRENT_DIR;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::OnceLock;
use tauri::command;

// create each constant
pub static ALLOWED_FILE_EXTENSIONS: OnceLock<HashSet<String>> = OnceLock::new();
pub static FAVOURITE_FILE_EXTENSIONS: OnceLock<HashMap<String, String>> = OnceLock::new();
pub static COPY_MODE: OnceLock<CopyMode> = OnceLock::new();
pub static NUMBER_RESULTS_LEVENHSTEIN: OnceLock<usize> = OnceLock::new();
pub static NUMBER_RESULTS_EMBEDDING: OnceLock<usize> = OnceLock::new();

#[derive(Debug, Deserialize)]
struct Settings {
    allowed_extensions: HashSet<String>,
    favourite_extensions: HashMap<String, String>,
    copy_mode: CopyMode,
    number_results_levenhstein: usize,
    number_results_embedding: usize,
}

#[derive(Debug, Deserialize)]
pub enum CopyMode {
    Clipboard,
    File,
}

impl Settings {
    fn default() -> Settings {
        let allowed_extensions: HashSet<String> = [
            // Text and documents
            "txt", "pdf", "doc", "docx", "rtf", "odt", "tex", "md", "epub",
            // Spreadsheets and presentations
            "xls", "xlsx", "csv", "ods", "ppt", "pptx", "odp", "key", // Images
            "jpg", "jpeg", "png", "gif", "bmp", "tiff", "svg", "webp", "ico", "raw",
            // Audio and video
            "mp3", "wav", "ogg", "flac", "aac", "wma", "m4a", "mp4", "avi", "mov", "wmv", "flv",
            "mkv", "webm", "m4v", "3gp", // Archives and data
            "zip", "rar", "7z", "tar", "gz", "bz2", "xz", "json", "xml", "yaml", "yml", "toml",
            "ini", "cfg", // Web and programming
            "html", "htm", "css", "js", "php", "asp", "jsp", "py", "java", "c", "cpp", "h", "hpp",
            "cs", "rs", "go", "rb", "pl", "swift", "kt", "ts", "coffee", "scala", "groovy", "lua",
            "r", // Scripts and executables
            "sh", "bash", "zsh", "fish", "bat", "cmd", "ps1", "exe", "dll", "so", "dylib",
            // Other formats
            "sql", "db", "sqlite", "mdb", "ttf", "otf", "woff", "woff2", "obj", "stl", "fbx", "dxf",
            "dwg", "psd", "ai", "ind", "iso", "img", "dmg", "bak", "tmp", "log", "pcap",
        ]
        .iter()
        .map(|&s| String::from(s))
        .collect();

        let favourite_extensions: HashMap<String, String> = [
            ("Images", "png,jpg,jpeg,gif"),
            ("Text", "txt,doc,docx,pdf,odt,rtf"),
            ("Video", "mp4,mp4a,avi"),
            (
                "Coding",
                "c,cpp,cs,java,js,html,css,php,py,rs,sh,swift,ts,xml",
            ),
        ]
        .iter()
        .map(|&(key, value)| (String::from(key), String::from(value)))
        .collect();

        Settings {
            allowed_extensions,
            favourite_extensions,
            copy_mode: CopyMode::File,
            number_results_levenhstein: 15,
            number_results_embedding: 25,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ColorConfig {
    background: String,
    font: String,
    table_head_background: String,
    table_every_second_row_background: String,
    table_border: String,
    input_border: String,
    button_hover: String,
    modal_background: String,
    modal_hover: String,
}

impl ColorConfig {
    pub fn default() -> ColorConfig {
        ColorConfig {
            background: String::from("#2f2f2f"),
            font: String::from("#f6f6f6"),
            table_head_background: String::from("#0f0f0f"),
            table_every_second_row_background: String::from("#1f1f1f"),
            table_border: String::from("#ccc"),
            input_border: String::from("#ccc"),
            button_hover: String::from("#ffffff"),
            modal_background: String::from("#1f1f1f"),
            modal_hover: String::from("#2f2f2f"),
        }
    }
}

fn read_config(config_path: &PathBuf) -> Result<String, ()> {
    // Open the file
    let mut file = match File::open(config_path) {
        Ok(file) => file,
        Err(_) => return Err(()),
    };

    // Read the file contents into a string
    let mut contents = String::new();
    match file.read_to_string(&mut contents) {
        Ok(_) => (),
        Err(_) => return Err(()),
    }
    Ok(contents)
}

pub fn initialize_config() -> Result<(), String> {
    // get the path to the config file
    let mut path = CURRENT_DIR.clone();
    path.push("data/config/config.json");

    // read the config file and parse it into the Settings struct (use default values when an error occurs)
    let config = match read_config(&path) {
        Ok(config) => serde_json::from_str(&config).unwrap_or_else(|_| Settings::default()),
        Err(_) => Settings::default(),
    };

    // set every constant, if something fails, the whole program immediantly stops executing
    match FAVOURITE_FILE_EXTENSIONS.set(config.favourite_extensions) {
        Ok(_) => {}
        Err(_) => return Err("couldn't set favourite extensions".to_string()),
    }
    match ALLOWED_FILE_EXTENSIONS.set(config.allowed_extensions) {
        Ok(_) => {}
        Err(_) => return Err("couldn't set allowed extensions".to_string()),
    }
    match COPY_MODE.set(config.copy_mode) {
        Ok(_) => {}
        Err(_) => return Err("couldn't set copy mode".to_string()),
    }
    match NUMBER_RESULTS_EMBEDDING.set(config.number_results_embedding) {
        Ok(_) => {}
        Err(_) => return Err("couldn't set num emb".to_string()),
    }
    match NUMBER_RESULTS_LEVENHSTEIN.set(config.number_results_levenhstein) {
        Ok(_) => {}
        Err(_) => return Err("couldn't set num lev".to_string()),
    }

    Ok(())
}

// functions for retrieving the values of the constants
#[command]
pub fn get_fav_file_extensions() -> HashMap<String, String> {
    match FAVOURITE_FILE_EXTENSIONS.get() {
        None => Settings::default().favourite_extensions,
        Some(val) => val.to_owned(),
    }
}

#[command]
pub fn get_allowed_file_extensions() -> HashSet<String> {
    match ALLOWED_FILE_EXTENSIONS.get() {
        None => Settings::default().allowed_extensions,
        Some(val) => val.to_owned(),
    }
}

pub fn get_copy_mode() -> CopyMode {
    match COPY_MODE.get() {
        None => Settings::default().copy_mode,
        Some(val) => match val {
            // needed to dereference it
            CopyMode::Clipboard => CopyMode::Clipboard,
            CopyMode::File => CopyMode::File,
        },
    }
}

pub fn get_number_results_levenhstein() -> usize {
    match NUMBER_RESULTS_LEVENHSTEIN.get() {
        None => Settings::default().number_results_levenhstein,
        Some(val) => val.to_owned(),
    }
}

pub fn get_number_results_embedding() -> usize {
    match NUMBER_RESULTS_EMBEDDING.get() {
        None => Settings::default().number_results_embedding,
        Some(val) => val.to_owned(),
    }
}

#[command]
pub fn get_css_settings() -> ColorConfig {
    // get the path to the color config file
    let mut path = CURRENT_DIR.clone();
    path.push("data/config/color-config.json");

    // read it contents and parse it to the struct, or use the default values
    match read_config(&path) {
        Ok(config) => serde_json::from_str(&config).unwrap_or_else(|_| ColorConfig::default()),
        Err(_) => ColorConfig::default(),
    }
}
