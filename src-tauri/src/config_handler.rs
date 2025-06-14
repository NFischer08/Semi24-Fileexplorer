use log::{error, info, warn};
use num_cpus;
use serde::{Deserialize, Serialize};
use serde_json;
use std::{
    collections::{HashMap, HashSet},
    env,
    fs::{self, File},
    io::Read,
    path::{absolute, PathBuf},
    sync::{LazyLock, OnceLock},
};
use tauri::command;

// create each constant
pub static ALLOWED_FILE_EXTENSIONS: OnceLock<HashSet<String>> = OnceLock::new();
pub static FAVOURITE_FILE_EXTENSIONS: OnceLock<HashMap<String, String>> = OnceLock::new();
pub static COPY_MODE: OnceLock<CopyMode> = OnceLock::new();
pub static NUMBER_RESULTS_LEVENSHTEIN: OnceLock<usize> = OnceLock::new();
pub static NUMBER_RESULTS_EMBEDDING: OnceLock<usize> = OnceLock::new();
pub static PATHS_TO_INDEX: OnceLock<Vec<PathBuf>> = OnceLock::new();
pub static INDEX_HIDDEN_FILES: OnceLock<bool> = OnceLock::new();
pub static INDEX_DIRECTORIES: OnceLock<bool> = OnceLock::new();
pub static INDEX_BINARIES: OnceLock<bool> = OnceLock::new();
pub static CREATE_BATCH_SIZE: OnceLock<usize> = OnceLock::new();
pub static SEARCH_BATCH_SIZE: OnceLock<usize> = OnceLock::new();
pub static NUMBER_OF_THREADS: OnceLock<usize> = OnceLock::new();
// Important Information: paths_to_ignore will still be watched by rt_db but not acted on
pub static PATHS_TO_IGNORE: OnceLock<Vec<PathBuf>> = OnceLock::new();
pub static PATH_TO_WEIGHTS: OnceLock<PathBuf> = OnceLock::new();
pub static PATH_TO_VOCAB: OnceLock<PathBuf> = OnceLock::new();
pub static EMBEDDING_DIMENSIONS: OnceLock<usize> = OnceLock::new();

// This should stay Lazy because it ensures that it can be used at all time
pub static CURRENT_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    env::current_dir().and_then(absolute).unwrap_or_else(|e| {
        error!("Failed to resolve absolute path: {}", e);
        PathBuf::from(".") // fallback to the current directory
    })
});

// create structs
#[derive(Debug, Deserialize)]
struct RawSettings {
    allowed_extensions: HashSet<String>,
    favourite_extensions: HashMap<String, String>,
    copy_mode: CopyMode,
    number_results_levenshtein: usize,
    number_results_embedding: usize,
    paths_to_index: Vec<String>,
    index_hidden_files: bool,
    index_directories: bool,
    index_binaries: bool,
    create_batch_size: usize,
    search_batch_size: usize,
    number_of_threads: usize,
    paths_to_ignore: Vec<String>,
    path_to_weights: String,
    path_to_vocab: String,
    embedding_dimensions: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Settings {
    allowed_extensions: HashSet<String>,
    favourite_extensions: HashMap<String, String>,
    copy_mode: CopyMode,
    number_results_levenshtein: usize,
    number_results_embedding: usize,
    paths_to_index: Vec<PathBuf>,
    index_hidden_files: bool,
    index_directories: bool,
    index_binaries: bool,
    create_batch_size: usize,
    search_batch_size: usize,
    number_of_threads: usize,
    paths_to_ignore: Vec<PathBuf>,
    path_to_weights: PathBuf,
    path_to_vocab: PathBuf,
    embedding_dimensions: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum CopyMode {
    Clipboard,
    File,
}
impl Default for Settings {
    /// creates some default values incase it's not able to read the JSON file properly
    fn default() -> Self {
        let allowed_extensions: HashSet<String> = [
            "txt", "pdf", "doc", "docx", "xls", "xlsx", "csv", "ppt", "pptx", "jpg", "jpeg", "png",
            "gif", "mp3", "wav", "mp4", "avi", "mov", "mkv", "zip", "rar", "7z", "tar", "gz",
            "html", "css", "js", "py", "java", "c", "cpp", "rs", "json", "xml", "sql", "log",
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
            number_results_levenshtein: 15,
            number_results_embedding: 25,
            paths_to_index: default_paths_to_index(),
            index_hidden_files: false,
            index_directories: true,
            index_binaries: true,
            create_batch_size: 250,
            search_batch_size: 2500,
            number_of_threads: num_cpus::get() - 1,
            paths_to_ignore: Vec::new(),
            path_to_weights: CURRENT_DIR.clone().join("data/model/eng_weights_D300"),
            path_to_vocab: CURRENT_DIR.clone().join("data/model/eng_vocab.json"),
            embedding_dimensions: 300,
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

impl Default for ColorConfig {
    fn default() -> Self {
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

/// creates the config file if it doesn't exist
pub fn build_config<T: serde::Serialize>(path: &PathBuf, settings: &T) -> bool {
    let json = match serde_json::to_string_pretty(&settings) {
        Ok(json) => json,
        Err(e) => {
            error!("Failed to serialize settings to JSON: {}", e);
            return false;
        }
    };

    if fs::write(path, json).is_err() {
        error!("Failed to write config file at {:?}", path);
        return false;
    }
    true
}

/// opens a file and returns its contents
fn read_config(config_path: &PathBuf) -> Result<String, ()> {
    // Open the file
    let mut file = match File::open(config_path) {
        Ok(file) => file,
        Err(e) => {
            error!("Failed to open config file {:?}: {}", config_path, e);
            return Err(());
        }
    };

    // Read the file contents into a string
    let mut contents = String::new();
    match file.read_to_string(&mut contents) {
        Ok(_) => (),
        Err(e) => {
            error!("Failed to read config file {:?}: {}", config_path, e);
            return Err(());
        }
    }
    info!("Successfully read config file {:?}", config_path);
    Ok(contents)
}

/// reads the config file and initializes the constants
pub fn initialize_config() {
    info!("Initializing config");
    let default_settings: Settings = Settings::default();
    // get the path to the config file
    let mut path: PathBuf = CURRENT_DIR.clone();
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
                        .map(PathBuf::from)
                        .filter(|path| path.exists())
                        .collect();
                    let paths_to_ignore: Vec<PathBuf> = settings
                        .paths_to_ignore
                        .iter()
                        .map(PathBuf::from)
                        .filter(|path| path.exists())
                        .collect();
                    Settings {
                        allowed_extensions: settings.allowed_extensions,
                        favourite_extensions: settings.favourite_extensions,
                        copy_mode: settings.copy_mode,
                        number_results_levenshtein: settings.number_results_levenshtein,
                        number_results_embedding: settings.number_results_embedding,
                        paths_to_index: if paths_to_index.is_empty() {
                            default_settings.paths_to_index
                        } else {
                            paths_to_index
                        },
                        index_hidden_files: settings.index_hidden_files,
                        index_directories: settings.index_directories,
                        index_binaries: settings.index_binaries,
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
                        embedding_dimensions: settings.embedding_dimensions,
                    }
                }
                Err(e) => {
                    warn!("Fehler beim Verarbeiten der Konfiguration ({}), es werden für manche Variablen Standardwerte verwendet.", e);
                    default_settings
                }
            }
        }
        Err(_) => {
            warn!("Fehler beim Verarbeiten der Konfiguration, es werden für manche Variablen Standardwerte verwendet.");
            default_settings
        }
    };

    // set every constant, if something fails, the whole program immediately stops executing due to panicking
    if let Err(e) = FAVOURITE_FILE_EXTENSIONS.set(config.favourite_extensions) {
        error!("Konnte favourite extensions nicht setzen: {:?}", e);
    }

    if let Err(e) = ALLOWED_FILE_EXTENSIONS.set(config.allowed_extensions) {
        error!("Konnte allowed extensions nicht setzen: {:?}", e);
    }

    if let Err(e) = COPY_MODE.set(config.copy_mode) {
        error!("Konnte copy mode nicht setzen: {:?}", e);
    }

    if let Err(e) = NUMBER_RESULTS_EMBEDDING.set(config.number_results_embedding) {
        error!("Konnte number_results_embedding nicht setzen: {:?}", e);
    }

    if let Err(e) = NUMBER_RESULTS_LEVENSHTEIN.set(config.number_results_levenshtein) {
        error!("Konnte number_results_levenshtein nicht setzen: {:?}", e);
    }

    if let Err(e) = PATHS_TO_INDEX.set(config.paths_to_index) {
        error!("Konnte paths_to_index nicht setzen: {:?}", e);
    }

    if let Err(e) = INDEX_HIDDEN_FILES.set(config.index_hidden_files) {
        error!("Konnte index_hidden_files nicht setzen: {:?}", e);
    }

    if let Err(e) = INDEX_DIRECTORIES.set(config.index_directories) {
        error!("Konnte index_directories nicht setzen: {:?}", e);
    }

    if let Err(e) = INDEX_BINARIES.set(config.index_binaries) {
        error!("Konnte index_binaries nicht setzen: {:?}", e);
    }

    if let Err(e) = CREATE_BATCH_SIZE.set(config.create_batch_size) {
        error!("Konnte create_batch_size nicht setzen: {:?}", e);
    }

    if let Err(e) = SEARCH_BATCH_SIZE.set(config.search_batch_size) {
        error!("Konnte search_batch_size nicht setzen: {:?}", e);
    }

    if let Err(e) = NUMBER_OF_THREADS.set(config.number_of_threads) {
        error!("Konnte number_of_threads nicht setzen: {:?}", e);
    }

    if let Err(e) = PATHS_TO_IGNORE.set(config.paths_to_ignore) {
        error!("Konnte paths_to_ignore nicht setzen: {:?}", e);
    }

    if let Err(e) = PATH_TO_WEIGHTS.set(config.path_to_weights) {
        error!("Konnte path_to_weights nicht setzen: {:?}", e);
    }

    if let Err(e) = PATH_TO_VOCAB.set(config.path_to_vocab) {
        error!("Konnte path_to_vocab nicht setzen: {:?}", e);
    }

    if let Err(e) = EMBEDDING_DIMENSIONS.set(config.embedding_dimensions) {
        error!("Konnte embedding_dimensions nicht setzen: {:?}", e);
    }
}

// functions for retrieving the values of the constants
#[command]
pub fn get_fav_file_extensions() -> HashMap<String, String> {
    match FAVOURITE_FILE_EXTENSIONS.get() {
        None => {
            print_warning("FAVOURITE_FILE_EXTENSIONS");
            Settings::default().favourite_extensions
        }
        Some(val) => val.to_owned(),
    }
}

#[command]
pub fn get_allowed_file_extensions() -> HashSet<String> {
    match ALLOWED_FILE_EXTENSIONS.get() {
        None => {
            print_warning("ALLOWED_FILE_EXTENSIONS");
            Settings::default().allowed_extensions
        }
        Some(val) => val.to_owned(),
    }
}

pub fn get_copy_mode() -> CopyMode {
    match COPY_MODE.get() {
        None => {
            print_warning("COPY_MODE");
            Settings::default().copy_mode
        }
        Some(val) => val.to_owned(),
    }
}

pub fn get_number_results_levenshtein() -> usize {
    match NUMBER_RESULTS_LEVENSHTEIN.get() {
        None => {
            print_warning("NUMBER_RESULTS_LEVENSHTEIN");
            Settings::default().number_results_levenshtein
        }
        Some(val) => val.to_owned(),
    }
}

pub fn get_number_results_embedding() -> usize {
    match NUMBER_RESULTS_EMBEDDING.get() {
        None => {
            print_warning("NUMBER_RESULTS_EMBEDDING");
            Settings::default().number_results_embedding
        }
        Some(val) => val.to_owned(),
    }
}

pub fn get_paths_to_index() -> Vec<PathBuf> {
    match PATHS_TO_INDEX.get() {
        None => {
            print_warning("PATHS_TO_INDEX");
            Settings::default().paths_to_index
        }
        Some(val) => val.to_owned(),
    }
}

pub fn get_create_batch_size() -> usize {
    match CREATE_BATCH_SIZE.get() {
        None => {
            print_warning("CREATE_BATCH_SIZE");
            Settings::default().create_batch_size
        }
        Some(val) => val.to_owned(),
    }
}

pub fn get_search_batch_size() -> usize {
    match SEARCH_BATCH_SIZE.get() {
        None => {
            print_warning("SEARCH_BATCH_SIZE");
            Settings::default().search_batch_size
        }
        Some(val) => val.to_owned(),
    }
}

pub fn get_number_of_threads() -> usize {
    match NUMBER_OF_THREADS.get() {
        None => {
            print_warning("NUMBER_OF_THREADS");
            Settings::default().number_of_threads
        }
        Some(val) => val.to_owned(),
    }
}

pub fn get_paths_to_ignore() -> Vec<PathBuf> {
    match PATHS_TO_IGNORE.get() {
        None => {
            print_warning("PATHS_TO_IGNORE");
            Settings::default().paths_to_ignore
        }
        Some(val) => val.to_owned(),
    }
}

pub fn get_path_to_weights() -> PathBuf {
    match PATH_TO_WEIGHTS.get() {
        None => {
            print_warning("PATH_TO_WEIGHTS");
            Settings::default().path_to_weights
        }
        Some(path) => path.to_owned(),
    }
}

pub fn get_path_to_vocab() -> PathBuf {
    match PATH_TO_VOCAB.get() {
        None => {
            print_warning("PATH_TO_VOCAB");
            Settings::default().path_to_vocab
        }
        Some(path) => path.to_owned(),
    }
}

pub fn get_embedding_dimensions() -> usize {
    match EMBEDDING_DIMENSIONS.get() {
        None => {
            print_warning("EMBEDDING_DIMENSIONS");
            Settings::default().embedding_dimensions
        }
        Some(val) => val.to_owned(),
    }
}

/// retrieves the CSS config settings to send them to the frontend
#[command]
pub fn get_css_settings() -> ColorConfig {
    // get the path to the color config file
    let mut path = CURRENT_DIR.clone();
    path.push("data/config/color-config.json");

    // read its contents and parse it to the struct, or use the default values
    match read_config(&path) {
        Ok(config) => serde_json::from_str(&config).unwrap_or_else(|e| {
            warn!("Failed to parse color-config.json: {}", e);
            ColorConfig::default()
        }),
        Err(_) => ColorConfig::default(),
    }
}

fn print_warning(var: &str) {
    warn!(
        "Die Variable '{}' konnte nicht gelesen werden, es wird auf den Standardwert zurückgegriffen.",
        var
    );
}

fn default_paths_to_index() -> Vec<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        vec![PathBuf::from("C:\\Users")]
    }

    #[cfg(target_os = "macos")]
    {
        vec![PathBuf::from("/Users")]
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        vec![PathBuf::from("/home")]
    }
}
