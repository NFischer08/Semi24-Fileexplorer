use crate::config_handler::get_paths_to_ignore;
use crate::manager::{initialize_globals, VOCAB, WEIGHTS};
use ndarray::{Array2, Axis};
use r2d2::PooledConnection;
use r2d2_sqlite::SqliteConnectionManager;
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::{
    collections::HashSet,
    fs::{self},
    path::{Path, PathBuf},
};

pub static PATHS_TO_IGNORE: LazyLock<Vec<PathBuf>> = LazyLock::new(get_paths_to_ignore);

#[derive(Debug, Clone)]
pub struct Files {
    pub(crate) id: i32,
    pub(crate) file_name: String,
    pub(crate) file_path: String,
    pub(crate) file_type: Option<String>,
}

/// Converts "\\" into "/" so that Windows and Unix systems have same path structure
pub fn convert_to_forward_slashes(path: &Path) -> String {
    path.to_str()
        .map(|s| s.replace('\\', "/"))
        .unwrap_or_else(String::new)
}

/// calculates the cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        panic!("cosine_similarity: input vectors must have the same length");
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
    for paths in PATHS_TO_IGNORE.clone() {
        if path == paths {
            return false;
        }
    }

    // Checks if the extension of the Path is in the allowed_file_extensions Hashset
    path.extension()
        .and_then(|s| s.to_str())
        .map(|ext| allowed_file_extensions.contains(ext))
        .unwrap_or(false)
}

/// Generates the Database if it doesn't already exists and makes sure that path is indexed
pub fn initialize_database(pooled_connection: &PooledConnection<SqliteConnectionManager>) {
    pooled_connection
        .pragma_update(None, "journal_mode", "WAL")
        .expect("journal_mode failed");
    pooled_connection
        .pragma_update(None, "synchronous", "NORMAL")
        .expect("synchronous failed");
    pooled_connection
        .pragma_update(None, "wal_autocheckpoint", "10000")
        .expect("wal_autocheckpoint failed");

    pooled_connection
        .execute(
            "CREATE TABLE IF NOT EXISTS files (
            id   INTEGER PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            name_embeddings BLOB NOT NULL
        )",
            (),
        )
        .expect("Could not create database");

    pooled_connection
        .execute(
            "CREATE INDEX IF NOT EXISTS idx_file_path ON files (file_path)",
            [],
        )
        .expect("Indexing failed: ");
}

/// Transforms Dates and Years into the Strings YEAR and DATE,
/// so that they don't take up space in Vocab, look at pytorch script
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

/// Loads the Vocab from vocab.json file
pub fn load_vocab(path: &PathBuf) -> HashMap<String, usize> {
    let vocab_json = fs::read_to_string(path).expect("Failed to read vocab file");
    serde_json::from_str(&vocab_json).expect("Failed to parse vocab JSON")
}

/// Reads the corresct Vecs from the weights matrix depending on the indices
pub fn embedding_from_ind(token_indices: Vec<usize>, weights: &Array2<f32>) -> Vec<f32> {
    let selected = weights.select(Axis(0), &token_indices);
    let sum_embedding = selected.sum_axis(Axis(0));
    let count = token_indices.len() as f32;
    let avg_embedding = &sum_embedding / count;
    avg_embedding.to_vec()
}

/// Makes embedding simple via using the other functions
pub fn full_emb(file_name: &str) -> Vec<f32> {
    initialize_globals(); // TODO
    let tokenized_file_name = tokenize_file_name(file_name);
    let indexed_file_name = tokens_to_indices(tokenized_file_name, VOCAB.get().unwrap());
    embedding_from_ind(indexed_file_name, WEIGHTS.get().unwrap())
}

/// Transforms the &Vec<u8> into Vec<f32> primary use case is to test
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
