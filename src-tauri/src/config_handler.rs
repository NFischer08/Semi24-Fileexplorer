use serde::Deserialize;
use serde_json;
use std::sync::OnceLock;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::ops::Deref;
use tauri::command;

pub static ALLOWED_FILE_EXTENSIONS: OnceLock<HashSet<String>> = OnceLock::new();
pub static FAVOURITE_FILE_EXTENSIONS: OnceLock<HashMap<String, String>> = OnceLock::new();
pub static COPY_MODE: OnceLock<CopyMode> = OnceLock::new();

#[derive(Debug, Deserialize)]
struct Settings {
    pub allowed_extensions: HashSet<String>,
    pub favourite_extensions: HashMap<String, String>,
    pub copy_mode: CopyMode,
}

#[derive(Debug, Deserialize)]
pub enum CopyMode {
    Clipboard,
    File
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
            copy_mode: CopyMode::File
        }
    }
}

fn read_config(config_path: &str) -> Settings {
    // Open the file
    let mut file = match File::open(config_path) {
        Ok(file) => file,
        Err(_) => return Settings::default(),
    };

    // Read the file contents into a string
    let mut contents = String::new();
    match file.read_to_string(&mut contents) {
        Ok(_) => (),
        Err(_) => return Settings::default(),
    }

    // Deserialize the JSON string into the Config struct
    serde_json::from_str(&contents).unwrap_or_else(|_| Settings::default())
}

pub fn initialize_config() -> Result<String, String> {
    let config = read_config(
        r"C:\Users\ninof\RustroverProjects\Semi24-Fileexplorer\src-tauri\src\config\config.json",
    ); // change path if needed
    println!("INIT: config: {:?}", config);
    match FAVOURITE_FILE_EXTENSIONS.set(config.favourite_extensions) {
        Ok(_) => {}
        Err(_) => return Err("couldn't set favourite extensions".to_string()),
    }
    match ALLOWED_FILE_EXTENSIONS.set(config.allowed_extensions) {
        Ok(_) => {}
        Err(_) => return Err("couldn't set allowed extensions".to_string()),
    }
    println!("INIT: ALLOWED_FILE_EXTENSIONS: {:?}", ALLOWED_FILE_EXTENSIONS.get());
    println!("INIT: FAVOURITE_FILE_EXTENSIONS: {:?}", FAVOURITE_FILE_EXTENSIONS.get());
    Ok("Properly set".to_string())
}

#[command]
pub fn get_fav_file_extensions() -> Result<HashMap<String, String>, String> {
    match FAVOURITE_FILE_EXTENSIONS.get() {
        Some(fav_ext) => Ok(fav_ext.to_owned()),
        None => Err("couldn't load value for some reason".to_string())
    }
}

#[command]
pub fn get_allowed_file_extensions() -> HashSet<String> {
    match ALLOWED_FILE_EXTENSIONS.get() {
        Some(ext) => ext.to_owned(),
        None => Settings::default().allowed_extensions
    }
}

pub fn get_copy_mode() -> CopyMode {
    match COPY_MODE.get() {
        Some(mode) => CopyMode::File, // TODO: fix Ownership and return mode
        None => CopyMode::File
    }
}