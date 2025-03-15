use r2d2::PooledConnection;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::prelude::*;
use rusqlite::Result;
use std::{
    collections::HashSet,
    fs::{self},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone)]
pub struct Files {
    pub(crate) id: i32,
    pub(crate) file_name: String,
    pub(crate) file_path: String,
    pub(crate) file_type: Option<String>,
}

pub fn convert_to_forward_slashes(path: &PathBuf) -> String {
    path.to_str()
        .map(|s| s.replace('\\', "/"))
        .unwrap_or_else(|| String::new())
}

pub fn cosine_similarity(search_embedding: &Vec<f32>, candidate_embedding: &Vec<f32>) -> f32 {
    let mut a2: f32 = 0.0;
    let mut b2: f32 = 0.0;
    let mut ab: f32 = 0.0;

    for (a, b) in search_embedding.iter().zip(candidate_embedding.iter()) {
        a2 += a * a;
        b2 += b * b;
        ab += a * b;
    }
    let result = ab / a2.sqrt() / b2.sqrt();
    result
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

pub fn check_database(
    mut conn: PooledConnection<SqliteConnectionManager>,
    allowed_file_extensions: &HashSet<String>,
    thread_pool: &rayon::ThreadPool,
) -> Result<(), Box<dyn std::error::Error>> {
    let bad_paths: Vec<String> = {
        let mut stmt = conn
            .prepare_cached("SELECT file_path FROM files")
            .expect("SQL statement preparing failed: ");
        let rows: Vec<Result<String, _>> = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .expect("Getting Rows failed: ")
            .collect();

        thread_pool.install(|| {
            rows.into_par_iter()
                .filter_map(|path_result| {
                    path_result.ok().and_then(|path| {
                        let path_obj = Path::new(&path);
                        if !is_allowed_file(path_obj, &allowed_file_extensions)
                            && !fs::metadata(&path).is_ok()
                        {
                            Some(path)
                        } else {
                            None
                        }
                    })
                })
                .collect()
        })
    };
    println!("Number of bad files: {}", bad_paths.len());

    if !bad_paths.is_empty() {
        let tx = conn.transaction().expect("Transaction creation failed");
        {
            let placeholders = bad_paths.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let query = format!("DELETE FROM files WHERE file_path IN ({})", placeholders);
            let mut stmt = tx
                .prepare_cached(&query)
                .expect("SQL statement preparing failed: ");
            stmt.execute(rusqlite::params_from_iter(bad_paths.iter()))
                .expect("SQL statement preparing failed: ");
        }
        tx.commit().expect("Transaction commit failed");
    }
    println!("Check Database completed");

    Ok(())
}

pub fn initialize_database(pooled_connection: &PooledConnection<SqliteConnectionManager>) -> () {
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

pub fn get_allowed_file_extensions() -> HashSet<String> {
    [
        // Text and documents
        "txt", "pdf", "doc", "docx", "rtf", "odt", "tex", "md", "epub",
        // Spreadsheets and presentations
        "xls", "xlsx", "csv", "ods", "ppt", "pptx", "odp", "key", // Images
        "jpg", "jpeg", "png", "gif", "bmp", "tiff", "svg", "webp", "ico", "raw",
        // Audio and video
        "mp3", "wav", "ogg", "flac", "aac", "wma", "m4a", "mp4", "avi", "mov", "wmv", "flv", "mkv",
        "webm", "m4v", "3gp", // Archives and data
        "zip", "rar", "7z", "tar", "gz", "bz2", "xz", "json", "xml", "yaml", "yml", "toml", "ini",
        "cfg", // Web and programming
        "html", "htm", "css", "js", "php", "asp", "jsp", "py", "java", "c", "cpp", "h", "hpp",
        "cs", "rs", "go", "rb", "pl", "swift", "kt", "ts", "coffee", "scala", "groovy", "lua", "r",
        // Scripts and executables
        "sh", "bash", "zsh", "fish", "bat", "cmd", "ps1", "exe", "dll", "so", "dylib",
        // Other formats
        "sql", "db", "sqlite", "mdb", "ttf", "otf", "woff", "woff2", "obj", "stl", "fbx", "dxf",
        "dwg", "psd", "ai", "ind", "iso", "img", "dmg", "bak", "tmp", "log", "pcap",
    ]
    .iter()
    .map(|&s| String::from(s))
    .collect()
}
