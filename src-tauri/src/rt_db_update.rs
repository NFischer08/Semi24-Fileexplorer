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
use std::{collections::HashSet, path::{Path, PathBuf}, sync::mpsc::channel, fs};
use r2d2::PooledConnection;
use r2d2_sqlite::SqliteConnectionManager;
use crate::manager::manager_make_pooled_connection;

pub fn start_file_watcher(watch_path: PathBuf, allowed_extensions: HashSet<String>) {
    // get the connection pool from manager
    let connection_pool = match manager_make_pooled_connection() {
        Ok(connection_pool) => connection_pool,
        Err(_) => {
            println!("Couldn't make pooled connection pool");
            return;
        }
    };

    // get a valid connection to db
    let pooled_connection: PooledConnection<SqliteConnectionManager> = match connection_pool.get() {
        Ok(pooled_connection) => pooled_connection,
        Err(_) => {
            println!("Couldn't get pooled connection from pool");
            return;
        }
    };

    // Create a channel to receive filesystem events
    let (sender, receiver) = channel::<notify::Result<Event>>();

    // Create a watcher and return if an error occurs
    let mut watcher = match recommended_watcher(sender) {
        Ok(watcher) => watcher,
        Err(e) => {
            println!("Error creating watcher: {:?}", e);
            return;
        }
    };

    // Start watching the specified path and return if an error occurs
    match watcher.watch(&watch_path, RecursiveMode::Recursive) {
        Ok(_) => {}
        Err(e) => {
            println!("Error watching path: {:?}", e);
            return;
        }
    }

    // creates a HashSet with paths to ignore
    let mut ignore: HashSet<&str> = HashSet::new();
    let mut trigger: bool = false;
    #[cfg(target_os = "windows")]
    {
        ignore.insert("$Recycle.Bin");
        ignore.insert("AppData");
        ignore.insert("Windows\\System32");
    }


    println!("Watching for changes in {:?}", watch_path);
    // Loop to receive events from the channel
    for res in receiver {
        match res {
            Ok(event) => {
                // usually only one path is returned (for-loop for safety)
                for file_path in &event.paths {
                    // ignore certain paths on windows
                    #[cfg(target_os = "windows")]
                    {
                        for folder in &ignore {
                            if file_path.to_string_lossy().contains(folder) {
                                trigger = true;
                                break;
                            }
                        }
                    }

                    // skip if path which needs to be ignored is found
                    if trigger {
                        trigger = false;
                        continue;
                    }

                    // check if the path is from interest
                    if file_path.is_dir() ||
                    file_path
                        .extension() // unpack extension and check if it is in the allowed extensions
                        .map(|ext| allowed_extensions.contains(&ext.to_string_lossy().to_string()))
                        .unwrap_or(false)
                    {
                        // get actual event kind and handle it
                        match event.kind {
                            Create(_) => {
                                // get needed values for db
                                let path = file_path.to_string_lossy();
                                let name = file_path.file_name().unwrap_or("ERR".as_ref()).to_string_lossy();
                                let file_type = file_path.extension().unwrap_or("ERR".as_ref()).to_string_lossy();

                                // insert file into db
                                match pooled_connection.execute("INSERT INTO files (file_name, file_path, file_type) VALUES (?, ?, ?)",
                                                          (path, name, file_type), ) {
                                    Ok(_) => {}
                                    Err(_) => println!("Error: Couldn't delete file into pooled connection"),
                                }

                                if file_path.is_dir() {
                                    todo!()
                                    // update db function starting at `file_path`
                                    // folder content is needed to be checked recursively
                                }
                            }
                            Remove(_) => {
                                println!("DELETE {:?}", file_path); // doesn't track folders for whatever reason :(
                                // remove file from db
                                match pooled_connection.execute("DELETE FROM files WHERE file_path = ?",
                                                          (file_path.to_string_lossy(),)) {
                                    Ok(_) => {},
                                    Err(_) => println!("Error: Couldn't delete file in pooled connection"),
                                }
                            }
                            Modify(modify) => match modify {
                                // only renaming is interesting
                                Name(mode) => {
                                    // From gives old path, To gives new path
                                    match mode {
                                        From => {
                                            // remove file from db
                                            match pooled_connection.execute("DELETE FROM files WHERE file_path = ?",
                                                                            (file_path.to_string_lossy(),)) {
                                                Ok(_) => {},
                                                Err(_) => println!("Error: Couldn't delete file in pooled connection"),
                                            }
                                        }
                                        To => {
                                            if file_path.is_dir() {
                                                let ppath = file_path.parent().unwrap_or(Path::new("/")).to_path_buf();
                                                check_folder(ppath, &pooled_connection).unwrap_or_default()
                                            } else {
                                                // get needed values
                                                let path = file_path.to_string_lossy();
                                                let name = file_path.file_name().unwrap_or("ERR".as_ref()).to_string_lossy();
                                                let file_type = file_path.extension().unwrap_or("ERR".as_ref()).to_string_lossy();
                                                // insert file into db
                                                match pooled_connection.execute("INSERT INTO files (file_name, file_path, file_type) VALUES (?, ?, ?)",
                                                                                (path, name, file_type), ) {
                                                    Ok(_) => {}
                                                    Err(_) => println!("Error: Couldn't delete file into pooled connection"),
                                                }
                                            }
                                        },
                                        // Linux: `Both` why? Idk, what it means? Idk :(
                                        _ => println!("Something else {:?}, ({:?})", file_path, mode) // proper implementing needed if it isnt a normal case
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

fn get_elements_in_dir(parent_path: PathBuf) -> Result<HashSet<PathBuf>, ()> {
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

fn check_folder(path: PathBuf, pooled_connection: &PooledConnection<SqliteConnectionManager>) -> Result<(), ()> {
    // read currently existing files in dir
    let mut current_files: HashSet<PathBuf> = match get_elements_in_dir(path) {
        Ok(paths) => paths,
        Err(_) => return Err(()),
    };

    // read currently existing files in db for that dir
    let mut db_files: HashSet<PathBuf> = HashSet::new(); // TODO: get all files and folders in dir

    // ignore the common elements
    let common_el: HashSet<PathBuf> = current_files.intersection(&db_files).cloned().collect();
    for el in common_el {
        current_files.remove(&el);
        db_files.remove(&el);
    }

    // files which now left in the `current_files` HashSet need to be inserted into the db since they are missing
    for file in current_files {
        let path = file.to_string_lossy();
        let name = file.file_name().unwrap_or("ERR".as_ref()).to_string_lossy();
        let file_type = file.extension().unwrap_or("ERR".as_ref()).to_string_lossy();
        match pooled_connection.execute("INSERT INTO files (file_name, file_path, file_type) VALUES (?, ?, ?)",
                                        (path, name, file_type), ) {
            Ok(_) => {}
            Err(_) => println!("Error: Couldn't delete file in pooled connection"),
        }
    }

    // files which are still in the db, but dont exist anymore need to be removed
    for file in db_files {
        match pooled_connection.execute("DELETE FROM files WHERE file_path = ?",
                                        (file.to_string_lossy(),)) {
            Ok(_) => {},
            Err(_) => println!("Error: Couldn't delete file in pooled connection"),
        }
    }

    Ok(())
}