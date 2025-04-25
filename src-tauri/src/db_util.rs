use r2d2::PooledConnection;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::Result;
use std::collections::HashMap;
use std::{
    collections::HashSet,
    fs::{self},
    path::{Path, PathBuf},
};
use std::ptr::read;
use std::time::Instant;
use bytemuck::{cast, cast_slice};
use regex::Regex;
use tauri::Error::CurrentDir;
use crate::manager::{CURRENT_DIR, VOCAB, WEIGHTS};
use ndarray::{Array2, Array1, Axis};

#[derive(Debug, Clone)]
pub struct Files {
    pub(crate) id: i32,
    pub(crate) file_name: String,
    pub(crate) file_path: String,
    pub(crate) file_type: Option<String>,
}

pub fn convert_to_forward_slashes(path: &Path) -> String {
    path.to_str()
        .map(|s| s.replace('\\', "/"))
        .unwrap_or_else(|| String::new())
}

pub fn cosine_similarity(
    search_embedding: &[f32],
    search_norm: f32,
    candidate_embedding: &[f32],
) -> f32 {
    let (b2, ab) = candidate_embedding
        .iter()
        .zip(search_embedding.iter())
        .fold((0.0, 0.0), |(b2, ab), (&b, &a)| (b2 + b * b, ab + a * b));

    ab / (search_norm * b2.sqrt())
}

pub fn is_allowed_file(path: &Path, allowed_file_extensions: &HashSet<String>) -> bool {
    if should_ignore_path(path) {
        return false;
    }
    path.extension()
        .and_then(|s| s.to_str())
        .map(|ext| allowed_file_extensions.contains(ext))
        .unwrap_or(false)
}

pub fn should_ignore_path(path: &Path) -> bool {
    path.to_str()
        .map_or(false, |s| s.starts_with("/proc") || s.starts_with("/sys"))
}

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
pub fn tokenize_file_name(file_name: &str) ->  Vec<String> {
    let re = Regex::new(r"[a-zäöü]+").unwrap();
    let matches: Vec<String> = re.find_iter(&file_name)
        .map(|mat| mat.as_str().to_string())
        .collect();
    matches
}

pub fn tokens_to_indices(tokens: Vec<String>, vocab: &HashMap<String, usize>) -> Vec<usize> {
    tokens
        .iter()
        .map(|token| *vocab.get(token).unwrap_or(&0)) // Default to index 0 for unknown tokens
        .collect()
}

pub fn load_vocab(path: &PathBuf) -> HashMap<String, usize> {
    let vocab_json = fs::read_to_string(path).expect("Failed to read vocab file");
    serde_json::from_str(&vocab_json).expect("Failed to parse vocab JSON")
}

pub fn embedding_from_ind(token_indices: Vec<usize>, embedding_dim: usize, vocab_len: usize) -> Vec<f32> {
    // Assume WEIGHTS is a OnceCell<Array2<f32>> and already initialized
    let weights: &Array2<f32> = WEIGHTS.get().unwrap();

    // Select embeddings for the given indices and sum them
    let sum_embedding: Array1<f32> = token_indices
        .iter()
        .map(|&idx| weights.row(idx))
        .fold(Array1::<f32>::zeros(embedding_dim), |acc, row| acc + &row);

    sum_embedding.to_vec()
}

pub fn full_emb(file_name: &str, embedding_dim: usize) -> Vec<f32> {
    let vocab = VOCAB.get().unwrap();
    let tokenized_file_name = tokenize_file_name(file_name);
    let indexed_file_name = tokens_to_indices(tokenized_file_name, vocab);
    embedding_from_ind(indexed_file_name, embedding_dim, vocab.len())
}
