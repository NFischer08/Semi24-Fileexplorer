use notify::event::ModifyKind::Name;
use notify::EventKind::{Create, Modify, Remove};
use notify::{self, recommended_watcher, Event, RecursiveMode, Result, Watcher};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::mpsc::channel;

pub fn start_file_watcher(path: PathBuf, allowed_extensions: HashSet<String>) {
    // Create a channel to receive filesystem events
    let (tx, rx) = channel::<Result<Event>>();

    // Create a watcher
    let mut watcher = recommended_watcher(tx).expect("Unable to create file watcher");

    // Start watching the specified path
    watcher
        .watch(&path, RecursiveMode::Recursive)
        .expect("Unable to watch file");

    println!("Watching for changes in {:?}", path);

    // Spawn a new thread to handle events
    for res in rx {
        match res {
            Ok(event) => {
                // Check if the event's paths contain any of the allowed extensions
                for file_path in &event.paths {
                    if let Some(extension) = file_path.extension() {
                        if allowed_extensions.contains(&extension.to_string_lossy().to_string()) {
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
                                    },
                                    _ => {}
                                },
                                _ => {}
                            }
                        }
                    }
                }
            }
            Err(e) => println!("watch error: {:?}", e),
        }
    }
    println!("File watcher stopped");
}
