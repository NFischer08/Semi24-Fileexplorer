use notify::{ self, recommended_watcher, Event, RecursiveMode, Watcher, EventKind::{ Create, Modify, Remove }, event::ModifyKind::Name };
use std::{collections::HashSet, path::PathBuf, sync::mpsc::channel};

pub fn start_file_watcher(path: PathBuf, allowed_extensions: HashSet<String>) {
    // Create a channel to receive filesystem events
    let (tx, rx) = channel::<notify::Result<Event>>();

    // Create a watcher and return if an error occurs
    let mut watcher = match recommended_watcher(tx) {
        Ok(watcher) => watcher,
        Err(e) => {
            println!("Error creating watcher: {:?}", e);
            return;
        }
    };

    // Start watching the specified path and return if an error occurs
    match watcher.watch(&path, RecursiveMode::Recursive) {
        Ok(_) => {}
        Err(e) => {
            println!("Error watching path: {:?}", e);
            return;
        }
    }

    println!("Watching for changes in {:?}", path);
    let mut mode: bool = true; // later used to determine if the file needs to be inserted or deleted when modified

    // Loop to receive events from the channel
    for res in rx {
        match res {
            Ok(event) => {
                // usually only one path is returned (for-loop for safety)
                for file_path in &event.paths {
                    // check if the path is from interest
                    if file_path.is_dir() || // TODO: Directories cause problems => scuffed when renaming, etc. => other handling needed?
                        file_path.extension() // unpack extension and check if it is in the allowed extensions
                            .map(|ext| allowed_extensions.contains(&ext.to_string_lossy().to_string()))
                            .unwrap_or(false) {
                        // get actual event kind and handle it
                        match event.kind {
                            Create(_) => {
                                println!("INSERT {:?}", file_path);
                            },
                            Remove(_) => {
                                println!("DELETE {:?}", file_path);
                            },
                            Modify(modify) => match modify {
                                Name(_) => {
                                    // TODO: Implement file modification handling
                                    // => 2 MODIFY events are triggered for each file modification
                                    // => First Path needs to be removed from the database, second Path needs to be added
                                    println!("Modify {:?}", file_path);
                                    // sol attempt !!DOES NOT WORK for folders!!
                                    if mode {
                                        println!("DELETE {:?}", file_path);
                                    } else {
                                        println!("INSERT {:?}", file_path);
                                    }
                                    mode = !mode;
                                },
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
