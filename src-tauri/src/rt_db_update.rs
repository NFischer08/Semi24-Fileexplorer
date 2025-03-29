use notify::{
    self,
    event::{
        ModifyKind::Name,
        RenameMode::{From, To},
    },
    recommended_watcher, Event, EventKind,
    EventKind::{Create, Modify, Remove},
    RecursiveMode, Watcher,
};
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
        Err(e) => { // TODO: bei OS Error nicht direkt returnen
            println!("Error watching path: {:?}", e);
            return;
        }
    }

    println!("Watching for changes in {:?}", path);
    // Loop to receive events from the channel
    for res in rx {
        match res {
            Ok(event) => {
                // usually only one path is returned (for-loop for safety)
                for file_path in &event.paths {
                    // check if the path is from interest
                    if file_path.is_dir() {
                        // go to parent folder and read every File / Dir in it (recursive)
                        match event.kind {
                            EventKind::Any => {println!("Any dir")}
                            EventKind::Access(kind) => {println!("Accessed dir {:?}", kind)}
                            Create(_) => {
                                println!("Create dir {:?}", file_path);
                            }
                            Modify(modifiii) => match modifiii {
                                Name(path) => println!("Renamed dir {:?}, {:?}", file_path, path),
                                _ => {}
                            },
                            Remove(x) => {
                                println!("Delete dir {:?} KIND: {:?}", file_path, x);
                            }
                            EventKind::Other => {println!("Other dir")}
                        }
                    } else if
                    //file_path.is_dir() || // TODO: Directories cause problems => scuffed when renaming, etc. => other handling needed?
                    file_path
                        .extension() // unpack extension and check if it is in the allowed extensions
                        .map(|ext| allowed_extensions.contains(&ext.to_string_lossy().to_string()))
                        .unwrap_or(false)
                    {
                        // get actual event kind and handle it
                        match event.kind {
                            Create(_) => {
                                println!("INSERT {:?}", file_path);
                            }
                            Remove(_) => {
                                println!("DELETE {:?}", file_path);
                            }
                            Modify(modify) => match modify {
                                // only renaming is interesting
                                Name(mode) => {
                                    // From gives old path, To gives new path
                                    match mode {
                                        From => println!("DELETE {:?}", file_path),
                                        To => println!("INSERT {:?}", file_path),
                                        _ => {} // proper implementing need if it isnt a normal case
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
