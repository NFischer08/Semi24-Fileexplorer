use r2d2::PooledConnection;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::Result;
use std::collections::HashMap;
use std::{
    collections::HashSet,
    fs::{self},
    path::{Path, PathBuf},
};
use tch::Tensor;

#[derive(Debug, Clone)]
pub struct Files {
    pub(crate) id: i32,
    pub(crate) file_name: String,
    pub(crate) file_path: String,
    pub(crate) file_type: Option<String>,
}

pub struct EmbeddingModel {
    pub embeddings: Tensor,
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
pub fn tokenize_file_name(file_name: &str) -> Vec<String> {
    // Split file name into tokens based on underscores and other delimiters
    file_name
        .split(['_', ' ', '-'])
        .map(|s| s.to_lowercase())
        .filter(|s| !s.is_empty()) // Remove empty tokens
        .collect()
}

pub fn load_vocab(path: &PathBuf) -> HashMap<String, usize> {
    let vocab_json = fs::read_to_string(path).expect("Failed to read vocab file");
    serde_json::from_str(&vocab_json).expect("Failed to parse vocab JSON")
}
pub fn tokens_to_indices(tokens: Vec<String>, vocab: &HashMap<String, usize>) -> Vec<usize> {
    tokens
        .iter()
        .map(|token| *vocab.get(token).unwrap_or(&0)) // Default to index 0 for unknown tokens
        .collect()
}
impl EmbeddingModel {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Load the embeddings from the saved PyTorch model
        let embeddings = Tensor::load(model_path)?;
        Ok(Self { embeddings })
    }

    pub fn get_embedding(&self, token_index: i64) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Retrieve the embedding for a specific token index
        if token_index >= self.embeddings.size()[0] {
            return Err("Token index out of bounds".into());
        }
        Ok(self.embeddings.get(token_index))
    }
}
