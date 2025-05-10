use crate::config_handler::{
    get_allowed_file_extensions, get_paths_to_ignore, get_paths_to_index, ALLOWED_FILE_EXTENSIONS,
};
use crate::db_util::full_emb;
use crate::manager::{manager_make_connection_pool, manager_populate_database};
use notify::{
    self,
    event::{
        ModifyKind::Name,
        RenameMode::{From, To},
    },
    recommended_watcher, Event,
    EventKind::{Create, Modify, Remove},
    RecursiveMode, Watcher,
};
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::mpsc::channel,
    thread,
};

/// gets all paths which need to be watched from config and starts watching each path
/// as well as that it initializes the db connection
pub fn start_file_watcher() {
    // creates a HashSet with paths to ignore
    let ignore: HashSet<String> = get_paths_to_ignore()
        .iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect();

    // get the connection pool from manager
    let connection_pool: Pool<SqliteConnectionManager> = manager_make_connection_pool();

    // start watching for changes in all paths to index
    for path in get_paths_to_index() {
        let conn = match connection_pool.get() {
            Ok(conn) => conn,
            Err(_) => continue,
        };
        let ignore = ignore.clone();
        thread::spawn(move || {
            watch_folder(
                path,
                &conn,
                &ignore.iter().map(|path| path.as_str()).collect(),
            )
        });
    }
}

/// watches a specific folder for changes and reports any changes to the db to keep it up to date
pub fn watch_folder(
    watch_path: PathBuf,
    pooled_connection: &PooledConnection<SqliteConnectionManager>,
    ignore: &HashSet<&str>,
) {
    
    println!("ignore: {:?}", ignore);
    
    let allowed_extensions: &HashSet<String> = match ALLOWED_FILE_EXTENSIONS.get() {
        Some(allowed_extensions) => allowed_extensions,
        None => &get_allowed_file_extensions(),
    };

    // Create a channel to receive filesystem events
    let (sender, receiver) = channel::<notify::Result<Event>>();

    // Create a watcher and panic if an error occurs
    let mut watcher = recommended_watcher(sender).expect("Error: Couldn't create watcher");

    // Start watching the specified path and panic if an error occurs
    if let Err(e) = watcher.watch(&watch_path, RecursiveMode::Recursive) {
        eprintln!("Error: Couldn't watch path {:?}: {}", watch_path, e); // If this happens we may have a problem, but if it panics here we have an even bigger problem
    }

    // Loop to receive events from the channel
    for res in receiver {
        match res {
            Ok(event) => {
                // usually only one path is returned (for-loop for safety)
                'event: for file_path in event.paths {
                    // ignore certain paths
                    for folder in ignore {
                        if file_path.to_string_lossy().contains(folder) {
                            continue 'event;
                        }
                    }

                    // check if the path is from interest
                    if file_path.is_dir()
                        || file_path
                            .extension() // unpack extension and check if it is in the allowed extensions
                            .map(|ext| {
                                allowed_extensions.contains(&ext.to_string_lossy().to_string())
                            })
                            .unwrap_or(false)
                    {
                        // get actual event kind and handle it
                        match event.kind {
                            Create(_) => {
                                // insert file into db
                                insert_into_db(pooled_connection, &file_path);

                                if file_path.is_dir() {
                                    // update db function starting at `file_path`
                                    // folder content is needed to be checked recursively
                                    let _ = manager_populate_database(file_path);
                                }
                            }
                            Remove(_) => {
                                delete_from_db(pooled_connection, &file_path);
                            }
                            Modify(modify) => match modify {
                                // only renaming is interesting
                                Name(mode) => {
                                    // From gives old path, To gives new path
                                    match mode {
                                        From => {
                                            // remove file from db
                                            delete_from_db(pooled_connection, &file_path);
                                        }
                                        To => {
                                            if file_path.is_dir() {
                                                // get parent path and check it
                                                let ppath = file_path
                                                    .parent()
                                                    .unwrap_or(Path::new("/"))
                                                    .to_path_buf();
                                                check_folder(ppath, pooled_connection)
                                                    .unwrap_or_default()
                                            } else {
                                                // insert file into db
                                                insert_into_db(pooled_connection, &file_path);
                                            }
                                        }
                                        // Other cases should not occur / are not from interest since they mean something don't go as planned
                                        // Cap on Linux creating txt files is other
                                        _ => {
                                            println!("Something else {:?}, ({:?})", file_path, mode)
                                        }
                                    }
                                }
                                _ => {}
                            },
                            _ => {}
                        }
                    }
                }
            }
            Err(e) => println!("watch error: {:?}", e),
        }
    }
    println!("File watcher stopped");
}

/// gets all elements from a given folder
fn get_elements_in_dir(parent_path: &PathBuf) -> Result<HashSet<PathBuf>, ()> {
    // create a new empty HashSet
    let mut elements: HashSet<PathBuf> = HashSet::new();

    // get all entries from parent folder
    let entries = match fs::read_dir(parent_path) {
        Ok(entries) => entries,
        Err(_) => return Err(()),
    };

    // iterate over each entry and add it to the HashSet
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        elements.insert(entry.path());
    }

    Ok(elements)
}

/// checks a folder for its contents and compares it with the db to keep it up to date
fn check_folder(
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
    let mut stmt = pooled_connection
        .prepare("SELECT file_path FROM files WHERE file_path LIKE ?1")
        .expect("Failed to prepare statement");

    let paths_iter = stmt
        .query_map(params![pattern], |row| row.get::<_, String>(0))
        .expect("Failed to get file paths");

    let db_files_all: Vec<PathBuf> = paths_iter
        .filter_map(Result::ok)
        .map(PathBuf::from)
        .collect();

    let mut db_files: HashSet<PathBuf> = HashSet::new();

    let path_slashes_amount: usize = path.components().count();
    for file in db_files_all {
        // Prüfe ob der Pfad tatsächlich ein Unterpfad des Elternpfads ist
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

    // files which are still in the db, but dont exist anymore need to be removed
    for file in db_files {
        let pattern = format!("{}%", file.to_str().unwrap().replace("\\", "/"));
        pooled_connection
            .execute("DELETE FROM files WHERE file_path LIKE ?1", (pattern,))
            .expect("Failed to execute statement");
    }

    Ok(())
}

/// deletes a given file path from the db (therefor taking connection to it)
pub fn delete_from_db(
    pooled_connection: &PooledConnection<SqliteConnectionManager>,
    file_path: &Path,
) {
    pooled_connection
        .execute(
            "DELETE FROM files WHERE file_path = ?",
            (file_path.to_string_lossy().replace("\\", "/"),),
        )
        .expect("Error: Couldn't delete file in pooled connection");
}

/// inserts a given file path into the db (therefor taking connection to it)
fn insert_into_db(
    pooled_connection: &PooledConnection<SqliteConnectionManager>,
    file_path: &Path,
) {
    let path = file_path.to_string_lossy().to_string().replace("\\", "/");
    let name = file_path
        .file_stem()
        .unwrap_or("ERR".as_ref())
        .to_string_lossy()
        .to_string();
    let file_type = Some(
        file_path
            .extension()
            .unwrap_or("ERR".as_ref())
            .to_string_lossy()
            .to_string(),
    );
    let embedding: Vec<u8> = full_emb(&name)
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    pooled_connection
        .execute(
            "INSERT INTO files (file_name, file_path, file_type, name_embeddings) VALUES (?, ?, ?, ?)",
            (name, path, file_type, embedding),
        )
        .expect("Error: Couldn't insert file in pooled connection");
}
