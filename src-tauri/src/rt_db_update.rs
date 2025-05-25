use crate::config_handler::{
    get_allowed_file_extensions, get_paths_to_ignore, get_paths_to_index, ALLOWED_FILE_EXTENSIONS,
    INDEX_HIDDEN_FILES,
};
use crate::db_util::{check_folder, delete_from_db, insert_into_db, is_allowed_file, is_hidden};
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
    println!("File Watcher started");
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
        eprintln!(
            "Error: Couldn't watch child path of {:?}: {}",
            watch_path, e
        ); // If this happens we may have a problem, but if it panics here we have an even bigger problem
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

                    if !*INDEX_HIDDEN_FILES.get().expect("") && is_hidden(&file_path) {
                        continue 'event;
                    }

                    // check if the path is from interest
                    if is_allowed_file(&file_path, allowed_extensions) {
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
                            Modify(Name(mode)) => {
                                // From gives old path, To gives new path
                                match mode {
                                    From => {
                                        // remove file from db
                                        delete_from_db(pooled_connection, &file_path);
                                    }
                                    To => {
                                        if file_path.is_dir() {
                                            // get parent path and check it
                                            let parent_path = file_path
                                                .parent()
                                                .unwrap_or(Path::new("/"))
                                                .to_path_buf();
                                            check_folder(parent_path, pooled_connection)
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
                        }
                    }
                }
            }
            Err(e) => println!("watch error: {:?}", e),
        }
    }
}

/// gets all elements from a given folder
pub fn get_elements_in_dir(parent_path: &PathBuf) -> Result<HashSet<PathBuf>, ()> {
    // get all entries from parent folder
    let entries = fs::read_dir(parent_path).map_err(|_| ())?;
    Ok(entries
        .into_iter()
        .filter(|entry| entry.is_ok())
        .map(|entry| {
            entry
                .expect("Found Err in entries, even though they were removed by a filter before")
                .path()
        })
        .filter(|path| {
            is_allowed_file(
                path,
                ALLOWED_FILE_EXTENSIONS
                    .get()
                    .unwrap_or(&get_allowed_file_extensions()),
            )
        })
        .collect())
}
