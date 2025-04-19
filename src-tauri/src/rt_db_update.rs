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
use crate::manager::manager_make_pooled_connection;

pub fn start_file_watcher(watch_path: PathBuf, allowed_extensions: HashSet<String>) {
    let connection_pool = match manager_make_pooled_connection() {
        Ok(connection_pool) => connection_pool,
        Err(_) => {
            println!("Couldn't make pooled connection pool");
            return;
        }
    };
    let pooled_connection = match connection_pool.get() {
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
                    #[cfg(target_os = "windows")] // for windows: ignore recylce bin
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

                                let path = file_path.to_string_lossy();
                                let name = file_path.file_name().unwrap_or("ERR".as_ref()).to_string_lossy();
                                let file_type = file_path.extension().unwrap_or("ERR".as_ref()).to_string_lossy();

                                println!("INSERT {:?}", file_path); // TODO: if folder: check content recursive (content doesn't get flagged else)
                                match pooled_connection.execute("INSERT INTO files (file_name, file_path, file_type) VALUES (?, ?, ?)",
                                                          (path, name, file_type), ) {
                                    Ok(_) => {}
                                    Err(_) => println!("Error: Couldn't delete file into pooled connection"),
                                }

                                if file_path.is_dir() {
                                    todo!()
                                    // update db function starting at `file_path`
                                }
                            }
                            Remove(_) => {
                                println!("DELETE {:?}", file_path); // doesn't track folders for whatever reason :(

                                match pooled_connection.execute("DELETE FROM files WHERE file_path = ?",
                                                          (file_path.to_string_lossy(),)) {
                                    Ok(_) => {},
                                    Err(_) => println!("Error: Couldn't delete file into pooled connection"),
                                }
                            }
                            Modify(modify) => match modify {
                                // only renaming is interesting
                                Name(mode) => {
                                    // From gives old path, To gives new path
                                    match mode {
                                        From => println!("N DELETE {:?}", file_path),
                                        To => {
                                            if file_path.is_dir() {
                                                let ppath = file_path.parent().unwrap_or(Path::new("/")).to_path_buf();
                                                check_folder(ppath).unwrap_or_default()
                                            } else {
                                                println!("N INSERT {:?}", file_path);
                                            }
                                        },
                                        // Linux: `Both` why? Idk, what it means? Idk :(
                                        _ => println!("Something else {:?}, ({:?})", file_path, mode) // proper implementing need if it isnt a normal case
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

fn get_folders_in_dir(parent_path: PathBuf) -> Result<HashSet<PathBuf>, ()> {
    let mut folders: HashSet<PathBuf> = HashSet::new();
    let entries = match fs::read_dir(parent_path) {
        Ok(entries) => entries,
        Err(_) => return Err(()),
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let path = entry.path();
        if path.is_dir() {
            folders.insert(path);
        }
    }
    Ok(folders)
}

fn check_folder(path: PathBuf) -> Result<(), ()> {
    let mut current_files: HashSet<PathBuf> = match get_folders_in_dir(path) {
        Ok(paths) => paths, //.into_iter().map(|path| path.to_string_lossy().to_string()).collect(), // converts HashSet with PathBufs to HashSet with String (where the paths only point to directories: filter(|path| path.is_dir()).)
        Err(_) => return Err(()),
    };
    let mut db_files: HashSet<PathBuf> = HashSet::new(); // TODO: get all files and folders in dir
    let common_el: HashSet<PathBuf> = current_files.intersection(&db_files).cloned().collect();
    for el in common_el {
        current_files.remove(&el);
        db_files.remove(&el);
    }

    for file in current_files {
        println!("C INSERT {:?}", file);
    }
    for file in db_files {
        println!("C DELETE {:?}", file);
    }

    Ok(())
}