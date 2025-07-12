use crate::config_handler::{
    get_paths_to_ignore, INDEX_BINARIES, INDEX_DIRECTORIES, INDEX_HIDDEN_FILES,
};
use crate::manager::{manager_populate_database, VOCAB, WEIGHTS};
use crate::rt_db_update::get_elements_in_dir;
use log::{error, info};
use ndarray::{Array2, Axis};
use r2d2::PooledConnection;
use r2d2_sqlite::SqliteConnectionManager;
use regex::Regex;
use rusqlite::params;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::{
    collections::HashSet,
    fs::{self},
    path::{Path, PathBuf},
};

pub static PATHS_TO_IGNORE: LazyLock<Vec<PathBuf>> = LazyLock::new(get_paths_to_ignore);

// const for Windows storing the Regex Expression for matching drive letters and the root path /
#[cfg(windows)]
pub static WINDOWS_FS_ROOTS: LazyLock<(Regex, &Path)> = LazyLock::new(|| {
    (Regex::new(r"^[A-Z]:(/|\\|\\\\)").unwrap(),
     Path::new("/"))
});

#[derive(Debug, Clone)]
pub struct Files {
    #[allow(dead_code)]
    pub(crate) id: i32,
    pub(crate) file_name: String,
    pub(crate) file_path: String,
    pub(crate) file_type: String,
}

/// Converts "\\" into "/" so that Windows and Unix systems have the same path structure
pub fn convert_to_forward_slashes(path: &Path) -> String {
    path.to_str()
        .map(|s| s.replace('\\', "/"))
        .unwrap_or_default()
}

/// calculates the cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        error!("Dimensions of database and current model don‘t match");
        return 0.0;
    }
    let dot = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f32>();
    let norm_a = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|&y| y * y).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Returns true if it is an allowed Path and false if the Path should be ignored
/// or the file extension should be ignored
pub fn is_allowed_file(path: &Path, allowed_file_extensions: &HashSet<String>) -> bool {
    for ignore_path in PATHS_TO_IGNORE.iter() {
        if path.starts_with(ignore_path) {
            return false;
        }
    }

    if !*INDEX_HIDDEN_FILES.get().unwrap_or_else(|| {
        error!("Failed to get INDEX_HIDDEN_FILES");
        &true
    }) && is_hidden(path)
    {
        return false;
    }

    if path.is_dir() {
        return *INDEX_DIRECTORIES.get().unwrap_or(&true);
    }

    path.extension()
        .and_then(|s| s.to_str())
        .map(|ext| allowed_file_extensions.contains(ext))
        .unwrap_or(*INDEX_BINARIES.get().unwrap_or(&true))
}

/// Generates the Database if it doesn't already exist and makes sure that path is indexed
pub fn initialize_database(pooled_connection: &PooledConnection<SqliteConnectionManager>) {
    if let Err(e) = pooled_connection.pragma_update(None, "journal_mode", "WAL") {
        error!("journal_mode konnte nicht gesetzt werden: {e}");
    }

    if let Err(e) = pooled_connection.pragma_update(None, "synchronous", "NORMAL") {
        error!("synchronous konnte nicht gesetzt werden: {}", e);
    }

    if let Err(e) = pooled_connection.pragma_update(None, "wal_autocheckpoint", "10000") {
        error!("wal_autocheckpoint konnte nicht gesetzt werden: {}", e);
    }

    if let Err(e) = pooled_connection.execute(
        "CREATE TABLE IF NOT EXISTS files (
            id   INTEGER PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            name_embeddings BLOB NOT NULL
        )",
        (),
    ) {
        error!("Could not create database: {}", e);
    }

    if let Err(e) = pooled_connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_file_path ON files (file_path)",
        [],
    ) {
        error!("Indexing failed: {}", e);
    }
}

/// Transforms Dates and Years into the Strings YEAR and DATE,
/// so that they don't take up space in Vocab, look at a pytorch script
pub fn normalize_token(token: &str) -> String {
    // Match YYYY-MM-DD, YYYY:MM:DD, YYYY.MM.DD
    let date_re = Regex::new(r"^\d{4}[-:.]\d{2}[-:.]\d{2}$").unwrap();
    let year_re = Regex::new(r"^\d{4}$").unwrap();

    if date_re.is_match(token) {
        "DATE".to_string()
    } else if year_re.is_match(token) {
        "YEAR".to_string()
    } else {
        token.to_string()
    }
}

/// Tokenizes the file names using the regex expression
pub fn tokenize_file_name(file_name: &str) -> Vec<String> {
    let file_name = file_name.to_lowercase();
    // Match words, dates, and years
    let re = Regex::new(r"[a-zäöü]+|\d{4}[-:.]\d{2}[-:.]\d{2}|\d{4}").unwrap();
    re.find_iter(&file_name)
        .map(|mat| normalize_token(mat.as_str()))
        .collect()
}

/// Maps the tokens to the indices in Vocab
pub fn tokens_to_indices(tokens: Vec<String>, vocab: &HashMap<String, usize>) -> Vec<usize> {
    tokens
        .iter()
        .map(|token| *vocab.get(token).unwrap_or(&0)) // 0 = <UNK>
        .collect()
}

/// Loads the Vocab from the vocab.json file
pub fn load_vocab(path: &PathBuf) -> HashMap<String, usize> {
    let vocab_json = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            error!("Failed to read vocab file: {}", e);
            return HashMap::new();
        }
    };
    serde_json::from_str(&vocab_json).unwrap_or_else(|e| {
        error!("Failed to parse vocab JSON: {}", e);
        HashMap::new()
    })
}

/// Reads the correct Vecs from the weight's matrix depending on the indices
pub fn embedding_from_ind(token_indices: Vec<usize>, weights: &Array2<f32>) -> Vec<f32> {
    let selected = weights.select(Axis(0), &token_indices);
    let sum_embedding = selected.sum_axis(Axis(0));
    let count = token_indices.len() as f32;
    let avg_embedding = &sum_embedding / count;
    avg_embedding.to_vec()
}

/// Makes embedding simple via using the other functions
pub fn full_emb(file_name: &str) -> Vec<f32> {
    let tokenized_file_name = tokenize_file_name(file_name);
    let indexed_file_name = tokens_to_indices(tokenized_file_name, VOCAB.get().unwrap());
    embedding_from_ind(indexed_file_name, WEIGHTS.get().unwrap())
}

/// Transforms the &Vec<u8> into Vec<f32> the primary use case is to test
/// if transformation between database and embedding is working correctly
pub fn bytes_to_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(arr)
        })
        .collect()
}

/// checks a folder for its contents and compares it with the db to keep it up to date
pub fn check_folder(
    path: PathBuf,
    pooled_connection: &PooledConnection<SqliteConnectionManager>,
) -> Result<(), ()> {
    // read currently existing files in dir
    let mut current_files: HashSet<PathBuf> = match get_elements_in_dir(&path) {
        Ok(paths) => paths,
        Err(_) => return Err(()),
    };

    // read currently existing files in db for that dir
    let pattern = format!("{}%", path.to_str().unwrap().replace("\\", "/"));
    let mut stmt =
        match pooled_connection.prepare("SELECT file_path FROM files WHERE file_path LIKE ?1") {
            Ok(stmt) => stmt,
            Err(e) => {
                error!("Failed to prepare statement: {}", e);
                return Err(());
            }
        };

    let paths_iter = match stmt.query_map(params![pattern], |row| row.get::<_, String>(0)) {
        Ok(iter) => iter,
        Err(e) => {
            error!("Failed to get file paths: {}", e);
            return Err(());
        }
    };

    let db_files_all: Vec<PathBuf> = paths_iter
        .filter_map(Result::ok)
        .map(PathBuf::from)
        .collect();

    let mut db_files: HashSet<PathBuf> = HashSet::new();

    let path_slashes_amount: usize = path.components().count();
    for file in db_files_all {
        // Check if Path is a child path
        if file.clone().components().count() == path_slashes_amount + 1 {
            db_files.insert(file);
        }
    }

    // ignore the common elements
    let common_el: HashSet<PathBuf> = current_files.intersection(&db_files).cloned().collect();
    for el in common_el {
        current_files.remove(&el);
        db_files.remove(&el);
    }

    // files which now left in the `current_files` HashSet need to be inserted into the db since they are missing
    for file in current_files {
        if file.is_dir() {
            let _ = manager_populate_database(file);
        } else {
            insert_into_db(pooled_connection, &file)
        }
    }

    // files which are still in the db but don't exist anymore need to be removed
    for file in db_files {
        let pattern = format!("{}%", file.to_str().unwrap().replace("\\", "/"));
        if let Err(e) =
            pooled_connection.execute("DELETE FROM files WHERE file_path LIKE ?1", (pattern,))
        {
            error!("Failed to execute statement: {}", e);
        }
    }

    Ok(())
}

/// deletes a given file path from the db (therefore taking connection to it)
pub fn delete_from_db(
    pooled_connection: &PooledConnection<SqliteConnectionManager>,
    file_path: &Path,
) {
    info!("Deleting {:?}", &file_path);

    let path_str = file_path.to_string_lossy().replace("\\", "/");
    /*
    if !path_str.ends_with('/') {
        path_str.push('/');
    }

     */
    let like_pattern = format!("{}%", path_str);

    if let Err(e) =
        pooled_connection.execute("DELETE FROM files WHERE file_path LIKE ?", (like_pattern,))
    {
        error!("Couldn't delete file in pooled connection: {}", e);
    }
}

/// inserts a given file path into the db (therefore taking connection to it)
pub fn insert_into_db(
    pooled_connection: &PooledConnection<SqliteConnectionManager>,
    file_path: &Path,
) {
    let path = file_path.to_string_lossy().to_string().replace("\\", "/");
    let name = file_path
        .file_stem()
        .unwrap_or("ERR".as_ref())
        .to_string_lossy()
        .to_string();
    let file_type = if file_path.is_dir() {
        String::from("dir")
    } else {
        file_path
            .extension()
            .unwrap_or("binary".as_ref())
            .to_string_lossy()
            .to_string()
    };
    let embedding: Vec<u8> = full_emb(&name)
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    if let Err(e) = pooled_connection.execute(
        "INSERT INTO files (file_name, file_path, file_type, name_embeddings) VALUES (?, ?, ?, ?)",
        (name, path, file_type, embedding),
    ) {
        error!("Couldn't insert file in pooled connection: {}", e);
    }
}

/// A functon for knowing if a folder or any parent is hidden for Unix Systems (macOS + Linux)
#[cfg(unix)]
pub fn is_hidden(path: &Path) -> bool {
    // Check if any component (except root) starts with a dot
    path.components().any(|comp| {
        comp.as_os_str()
            .to_str()
            .is_some_and(|s| s.starts_with('.'))
    })
}

/// A functon for knowing if a folder is hidden for Windows also check if any parent folder is hidden
#[cfg(windows)]
pub fn is_hidden(path: &Path) -> bool {
    use std::fs;
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_HIDDEN: u32 = 0x2;
    let (re_root, slash_path) = WINDOWS_FS_ROOTS.clone();

    for ancestor in path.ancestors() {
        if let Ok(metadata) = fs::metadata(ancestor) {
            if (re_root.is_match(&ancestor.to_string_lossy().to_string()) && ancestor.to_string_lossy().len() <= 4) || ancestor == slash_path {
                break;
            }
            if (metadata.file_attributes() & FILE_ATTRIBUTE_HIDDEN) != 0 {
                return true;
            }
        }
    }
    false
}