use num_cpus;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::{absolute, PathBuf};
use std::sync::{LazyLock, OnceLock};
use tauri::command;

// create each constant
pub static ALLOWED_FILE_EXTENSIONS: OnceLock<HashSet<String>> = OnceLock::new();
pub static FAVOURITE_FILE_EXTENSIONS: OnceLock<HashMap<String, String>> = OnceLock::new();
pub static COPY_MODE: OnceLock<CopyMode> = OnceLock::new();
pub static NUMBER_RESULTS_LEVENHSTEIN: OnceLock<usize> = OnceLock::new();
pub static NUMBER_RESULTS_EMBEDDING: OnceLock<usize> = OnceLock::new();
pub static SEARCH_WITH_MODEL: OnceLock<bool> = OnceLock::new();
pub static PATHS_TO_INDEX: OnceLock<Vec<PathBuf>> = OnceLock::new();
pub static CREATE_BATCH_SIZE: OnceLock<usize> = OnceLock::new();
pub static SEARCH_BATCH_SIZE: OnceLock<usize> = OnceLock::new();
pub static NUMBER_OF_THREADS: OnceLock<usize> = OnceLock::new();
pub static PATHS_TO_IGNORE: OnceLock<Vec<PathBuf>> = OnceLock::new();
pub static PATH_TO_WEIGHTS: OnceLock<PathBuf> = OnceLock::new();
pub static PATH_TO_VOCAB: OnceLock<PathBuf> = OnceLock::new();
// This should stay Lazy because it ensures that it can be used at all time
pub static CURRENT_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    env::current_dir()
        .and_then(absolute)
        .expect("Failed to resolve absolute path")
    // VERY IMPORTANT when using .push() don't start with a /, if you do it will override the path with C: + "Your Input"
});

// create structs
#[derive(Debug, Deserialize)]
struct RawSettings {
    allowed_extensions: HashSet<String>,
    favourite_extensions: HashMap<String, String>,
    copy_mode: CopyMode,
    number_results_levenhstein: usize,
    number_results_embedding: usize,
    search_with_model: bool,
    paths_to_index: Vec<String>,
    create_batch_size: usize,
    search_batch_size: usize,
    number_of_threads: usize,
    paths_to_ignore: Vec<String>,
    path_to_weights: String,
    path_to_vocab: String,
}

#[derive(Clone, Debug)]
struct Settings {
    allowed_extensions: HashSet<String>,
    favourite_extensions: HashMap<String, String>,
    copy_mode: CopyMode,
    number_results_levenhstein: usize,
    number_results_embedding: usize,
    search_with_model: bool,
    paths_to_index: Vec<PathBuf>,
    create_batch_size: usize,
    search_batch_size: usize,
    number_of_threads: usize,
    paths_to_ignore: Vec<PathBuf>,
    path_to_weights: PathBuf,
    path_to_vocab: PathBuf,
}

#[derive(Debug, Deserialize, Clone)]
pub enum CopyMode {
    Clipboard,
    File,
}

impl Settings {
    /// creates some default values incase its not able to read the json file properly
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
            search_with_model: false,
            paths_to_index: vec![PathBuf::from("/")],
            create_batch_size: 250,
            search_batch_size: 1000,
            number_of_threads: num_cpus::get() - 1,
            paths_to_ignore: Vec::new(),
            path_to_weights: CURRENT_DIR.clone().join("data/model/weights"),
            path_to_vocab: CURRENT_DIR.clone().join("data/model/vocab.json"),
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

/// opens a file and returns it contents
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
    println!("Successfully read config file");
    Ok(contents)
}

/// reads the config file and initialises the constants
pub fn initialize_config() {
    println!("Initializing config");
    let default_settings = Settings::default();
    // get the path to the config file
    let mut path = CURRENT_DIR.clone();
    path.push("data/config/config.json");

    // read the config file and parse it into the Settings struct (use default values when an error occurs)
    let config: Settings = match read_config(&path) {
        Ok(config) => {
            let raw_settings: serde_json::error::Result<RawSettings> =
                serde_json::from_str(&config);
            // parse the raw settings to real Settings
            match raw_settings {
                Ok(settings) => {
                    let path_to_weights: PathBuf = PathBuf::from(settings.path_to_weights);
                    let path_to_vocab: PathBuf = PathBuf::from(settings.path_to_vocab);
                    let paths_to_index: Vec<PathBuf> = settings
                        .paths_to_index
                        .iter()
                        .map(|path| PathBuf::from(path))
                        .filter(|path| path.exists())
                        .collect();
                    let paths_to_ignore: Vec<PathBuf> = settings
                        .paths_to_ignore
                        .iter()
                        .map(|path| PathBuf::from(path))
                        .filter(|path| path.exists())
                        .collect();

                    Settings {
                        allowed_extensions: settings.allowed_extensions,
                        favourite_extensions: settings.favourite_extensions,
                        copy_mode: settings.copy_mode,
                        number_results_levenhstein: settings.number_results_levenhstein,
                        number_results_embedding: settings.number_results_embedding,
                        search_with_model: settings.search_with_model,
                        paths_to_index: if paths_to_index.is_empty() {
                            default_settings.paths_to_index
                        } else {
                            paths_to_index
                        },
                        create_batch_size: settings.create_batch_size,
                        search_batch_size: settings.search_batch_size,
                        number_of_threads: settings.number_of_threads,
                        paths_to_ignore: if paths_to_ignore.is_empty() {
                            default_settings.paths_to_ignore
                        } else {
                            paths_to_ignore
                        },
                        path_to_weights: if path_to_weights.exists() {
                            path_to_weights
                        } else {
                            default_settings.path_to_weights
                        },
                        path_to_vocab: if path_to_vocab.exists() {
                            path_to_vocab
                        } else {
                            default_settings.path_to_vocab
                        },
                    }
                }
                Err(_) => default_settings,
            }
        }
        Err(_) => default_settings,
    };

    // set every constant, if something fails, the whole program immediantly stops executing due to panicing
    FAVOURITE_FILE_EXTENSIONS
        .set(config.favourite_extensions)
        .expect("couldn't set favourite extensions");

    ALLOWED_FILE_EXTENSIONS
        .set(config.allowed_extensions)
        .expect("couldn't set allowed extensions");

    COPY_MODE
        .set(config.copy_mode)
        .expect("couldn't set copy mode");

    NUMBER_RESULTS_EMBEDDING
        .set(config.number_results_embedding)
        .expect("couldn't set num emb");

    NUMBER_RESULTS_LEVENHSTEIN
        .set(config.number_results_levenhstein)
        .expect("couldn't set num lev");

    SEARCH_WITH_MODEL
        .set(config.search_with_model)
        .expect("couldn't set search with model");

    PATHS_TO_INDEX
        .set(config.paths_to_index)
        .expect("couldn't set paths to index");

    CREATE_BATCH_SIZE
        .set(config.create_batch_size)
        .expect("couldn't set create batch size");

    SEARCH_BATCH_SIZE
        .set(config.search_batch_size)
        .expect("couldn't set search batch size");

    NUMBER_OF_THREADS
        .set(config.number_of_threads)
        .expect("couldn't set number of threads");

    PATHS_TO_IGNORE
        .set(config.paths_to_ignore)
        .expect("couldn't set paths to ignore");

    PATH_TO_WEIGHTS
        .set(config.path_to_weights)
        .expect("couldn't set path to weights");

    PATH_TO_VOCAB
        .set(config.path_to_vocab)
        .expect("couldn't set path to vocab");
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
        Some(val) => val.to_owned(),
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

pub fn get_search_with_model() -> bool {
    match SEARCH_WITH_MODEL.get() {
        None => Settings::default().search_with_model,
        Some(val) => val.to_owned(),
    }
}

pub fn get_paths_to_index() -> Vec<PathBuf> {
    match PATHS_TO_INDEX.get() {
        None => Settings::default().paths_to_index,
        Some(val) => val.to_owned(),
    }
}

pub fn get_create_batch_size() -> usize {
    match CREATE_BATCH_SIZE.get() {
        None => Settings::default().create_batch_size,
        Some(val) => val.to_owned(),
    }
}

pub fn get_search_batch_size() -> usize {
    match SEARCH_BATCH_SIZE.get() {
        None => Settings::default().search_batch_size,
        Some(val) => val.to_owned(),
    }
}

pub fn get_number_of_threads() -> usize {
    match NUMBER_OF_THREADS.get() {
        None => Settings::default().number_of_threads,
        Some(val) => val.to_owned(),
    }
}

pub fn get_paths_to_ignore() -> Vec<PathBuf> {
    match PATHS_TO_IGNORE.get() {
        None => Settings::default().paths_to_ignore,
        Some(val) => val.to_owned(),
    }
}

pub fn get_path_to_weights() -> PathBuf {
    match PATH_TO_WEIGHTS.get() {
        None => Settings::default().path_to_weights,
        Some(path) => path.to_owned(),
    }
}

pub fn get_path_to_vocab() -> PathBuf {
    match PATH_TO_VOCAB.get() {
        None => Settings::default().path_to_vocab,
        Some(path) => path.to_owned(),
    }
}

/// retireves the css config settings to send them to the frontend
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
