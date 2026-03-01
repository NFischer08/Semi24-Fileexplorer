use crate::config_handler::{get_search_batch_size, EMBEDDING_DIMENSIONS};
use crate::db_util::{cosine_similarity, full_emb, tokenize_file_name, tokens_to_indices,
};
use crate::file_information::FileDataFormatted;
use crate::manager::{AppState, VOCAB,
};
use log::{error, info};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rayon::iter::ParallelIterator;
use rayon::iter::{ParallelBridge};
use rayon::prelude::{ParallelSliceMut};
use std::cmp::Ordering;
use std::iter::repeat_n;
use std::sync::Arc;
use std::{
    fs::{self},
    path::{Path, PathBuf},
    time::Instant,
};
use strsim::normalized_levenshtein;
use tauri::{Emitter, State};

#[derive(Clone, Debug)]
pub struct FlatBatch {
    pub paths: Vec<String>,
    pub embeddings: Vec<f32>,
    pub names: Vec<String>,
}

/// Searches for similar File names in the Database via Levenshtein and a custom skip-gram model,
/// it uses connection_pool, search_term, search_path, search_file_type, num_results_lev, num_results_emb and state

#[tauri::command]
pub fn search_database(
    connection_pool: Pool<SqliteConnectionManager>,
    search_term: &str,
    search_path: PathBuf,
    search_file_types: String,
    num_results_embeddings: usize,
    num_results_levenshtein: usize,
    state: State<AppState>,
) {
    let handle = state.handle.clone();
    run_search_logic(
        connection_pool,
        search_term,
        search_path,
        search_file_types,
        num_results_embeddings,
        num_results_levenshtein,
        move |results| {
            let _ = handle.emit("search-finished", &results);
        },
    );
}

pub fn run_search_logic(
    connection_pool: Pool<SqliteConnectionManager>,
    search_term: &str,
    search_path: PathBuf,
    search_file_types: String,
    num_results_embeddings: usize,
    num_results_levenshtein: usize,
    // Use a closure instead of state.handle.emit
    on_progress: impl Fn(Vec<FileDataFormatted>) + Send + Sync + 'static,
) {
    // Getting a Pooled Connection
    match connection_pool.get() {
        Ok(conn) => conn,
        Err(e) => {
            error!("Failed to get connection pool: {e}");
            return;
        }
    };
    let batch_size: usize = get_search_batch_size();

    let start_time = Instant::now();

    //Setting Search Path to "" for searching everything
    let search_path_str = if cfg!(windows) && search_path.to_str().unwrap_or("") == "/" {
        String::new()
    } else {
        search_path.to_str().unwrap_or("").to_string()
    };

    //Making sure there are no Spaces in file_types and also accounting for "."
    let search_file_types_vec: Vec<String> = search_file_types
        .replace(" ", "")
        .replace(".", "")
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    info!("search file type: {search_file_types:?}");

    let sql_stmt: String = if search_file_types_vec.is_empty() {
        r#"
    SELECT file_path, file_name, name_embeddings
    FROM files
    WHERE file_path LIKE ?1
    "#
        .to_string()
    } else {
        let placeholders = repeat_n("?", search_file_types_vec.len())
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            r#"
        SELECT file_path, file_name, name_embeddings
        FROM files
        WHERE file_path LIKE ?1
        AND file_type IN ({placeholders})
        "#
        )
    };

    //Creating channel
    let (sender, receiver) = crossbeam_channel::bounded(batch_size * 2);

    // 1. Start the Query Thread with Zero-Copy Casting
    let query_thread = std::thread::spawn(move || {
        let mut pooled_connection = match connection_pool.get() {
            Ok(conn) => conn,
            Err(e) => {
                error!("Pool error: {e}");
                return;
            }
        };

        let tx = match pooled_connection.transaction() {
            Ok(tx) => tx,
            Err(e) => {
                error!("TX error: {e}");
                return;
            }
        };

        let search_pattern = format!("{search_path_str}%");
        let mut params: Vec<&dyn rusqlite::ToSql> =
            Vec::with_capacity(1 + search_file_types_vec.len());
        params.push(&search_pattern);
        for file_type in &search_file_types_vec {
            params.push(file_type);
        }

        {
            let mut stmt = match tx.prepare_cached(&sql_stmt) {
                Ok(stmt) => stmt,
                Err(e) => {
                    error!("Stmt error: {e}");
                    return;
                }
            };

            let mapped = match stmt.query_map(params.as_slice(), |row| {
                let path: String = row.get(0)?;
                let name: String = row.get(1)?;
                let bytes: Vec<u8> = row.get(2)?;
                Ok((path, name, bytes))
            }) {
                Ok(mapped) => mapped,
                Err(e) => {
                    error!("Query error: {e}");
                    return;
                }
            };

            let dim = *EMBEDDING_DIMENSIONS.get().unwrap_or(&300);
            let mut current_paths = Vec::with_capacity(batch_size);
            let mut current_names = Vec::with_capacity(batch_size);
            let mut current_embs = Vec::with_capacity(batch_size * dim);

            for (path, name, bytes) in mapped.flatten() {
                if bytes.len() == dim * 4 {
                    let f32_slice: &[f32] = bytemuck::cast_slice(&bytes);

                    current_paths.push(path);
                    current_names.push(name);
                    current_embs.extend_from_slice(f32_slice);

                    if current_paths.len() >= batch_size {
                        let _ = sender.send(FlatBatch {
                            paths: std::mem::take(&mut current_paths),
                            names: std::mem::take(&mut current_names),
                            embeddings: std::mem::take(&mut current_embs),
                        });
                        current_paths = Vec::with_capacity(batch_size);
                        current_names = Vec::with_capacity(batch_size);
                        current_embs = Vec::with_capacity(batch_size * dim);
                    }
                }
            }

            if !current_paths.is_empty() {
                let _ = sender.send(FlatBatch {
                    paths: current_paths,
                    names: current_names,
                    embeddings: current_embs,
                });
            }
        }
        let _ = tx.commit();
    });

    // 2. Pre-calculate search embedding
    let embedded_vec_f32 = full_emb(search_term);
    let dim = *EMBEDDING_DIMENSIONS.get().unwrap_or(&300);

    // 3. Parallel Processing with Arc-sharing
    let (results_lev, results_emb): (Vec<_>, Vec<_>) = receiver
        .into_iter()
        .par_bridge()
        .fold(
            || (Vec::new(), Vec::new()),
            |mut acc, batch| {
                let count = batch.paths.len();
                acc.0.reserve(count);
                acc.1.reserve(count);

                for ((path, name), emb_slice) in batch
                    .paths
                    .into_iter()
                    .zip(batch.names.into_iter())
                    .zip(batch.embeddings.chunks_exact(dim))
                {
                    let p_arc = Arc::new(path);
                    let n_arc = Arc::new(name);

                    // Calculations remain "hot" in cache
                    let lev = normalized_levenshtein(&n_arc, search_term) as f32;
                    let cos = cosine_similarity(emb_slice, &embedded_vec_f32);

                    acc.0.push((Arc::clone(&p_arc), Arc::clone(&n_arc), lev));
                    acc.1.push((p_arc, n_arc, cos));
                }
                acc
            },
        )
        .reduce(
            || (Vec::new(), Vec::new()),
            |mut acc1, mut acc2| {
                acc1.0.append(&mut acc2.0);
                acc1.1.append(&mut acc2.1);
                acc1
            },
        );

    // 4. Final Metadata Retrieval (Metadata only for top N results)
    let ret_lev = build_results(results_lev, num_results_levenshtein);
    on_progress(ret_lev.clone());

    let _ = query_thread.join();

    let mut final_num_emb = num_results_embeddings;
    if let Some(vocab) = VOCAB.get() {
        let tok = tokenize_file_name(search_term);
        if tokens_to_indices(tok, vocab).iter().all(|i| *i == 0) {
            final_num_emb = 0;
        }
    }

    let ret_emb = build_results(results_emb, final_num_emb);
    let mut final_results = ret_lev;
    final_results.extend(ret_emb);

    on_progress(final_results);

    info!("Search finished in {:?}", start_time.elapsed());
}

/// Support Function for searching which only gives back the best results in the form of DirEntries
fn build_results(
    mut matches: Vec<(Arc<String>, Arc<String>, f32)>,
    num_ret: usize,
) -> Vec<FileDataFormatted> {
    // Sort on contiguous score data
    matches.par_sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
    matches.truncate(num_ret);

    matches
        .into_iter()
        .map(|(path_arc, name_arc, _score)| {
            let path = Path::new(path_arc.as_str());
            let metadata = fs::metadata(path).ok();

            FileDataFormatted {
                name: name_arc.as_ref().clone(),
                path: path_arc.as_ref().clone(),
                last_modified: metadata
                    .as_ref()
                    .and_then(|m| m.modified().ok())
                    .map(|t| {
                        let datetime: chrono::DateTime<chrono::Local> = t.into();
                        datetime.format("%d.%m.%Y %H:%M").to_string()
                    })
                    .unwrap_or_else(|| "Unknown".to_string()),
                size: metadata
                    .map(|m| format!("{} KB", m.len() / 1024))
                    .unwrap_or_else(|| "0 KB".to_string()),
                file_type: path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("file")
                    .to_string(),
            }
        })
        .collect()
}
