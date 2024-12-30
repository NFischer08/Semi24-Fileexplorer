use std::{
    sync::mpsc::{channel},
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};
use strsim::normalized_levenshtein;
use walkdir::WalkDir;

/// Finds exact matches for the search term while building a Vec of entries for further processing.
///
/// Returns a tuple containing the total entries checked during the exact search and the Vec of entries.
fn find_exact_matches_parallel_and_collect(
    path: &str,
    search_term: &str,
) -> (u32, Vec<walkdir::DirEntry>) {
    let start_time = Instant::now(); // Start measuring time
    let count = Arc::new(Mutex::new(0)); // Shared counter for all entries
    let collected_entries = Arc::new(Mutex::new(Vec::new())); // Shared collection for entries

    let (tx, rx) = channel(); // Channel for communication between threads
    let mut threads = Vec::new();

    // Collect all top-level directories in the path
    for entry in WalkDir::new(path).max_depth(1).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_dir() {
            let dir_path = entry.path().to_path_buf();
            let tx_clone = tx.clone();
            let count_clone = Arc::clone(&count);
            let collected_entries_clone = Arc::clone(&collected_entries);
            let search_term = search_term.to_owned();

            // Spawn a thread for each directory
            let handle = thread::spawn(move || {
                for sub_entry in WalkDir::new(dir_path.clone()).into_iter().filter_map(|e| e.ok()) {
                    let file_name = sub_entry.file_name().to_string_lossy();

                    // Increment the counter
                    *count_clone.lock().unwrap() += 1;

                    // Save the entry for later similarity search
                    collected_entries_clone.lock().unwrap().push(sub_entry.clone());

                    // Check for exact match
                    if file_name == search_term {
                        let full_path = sub_entry.path().to_string_lossy().into_owned();
                        tx_clone.send(full_path).unwrap(); // Send the result to the main thread
                    }
                }

                println!("Finished processing directory: {:?}", dir_path.display());
            });

            threads.push(handle);
        }
    }

    // Start a thread to print results received via the channel
    let printer = thread::spawn(move || {
        for message in rx {
            let elapsed_time = start_time.elapsed();
            println!("Found exact match: {}", message);
            println!("Time elapsed: {:.2?}", elapsed_time);
        }
    });

    // Wait for all threads to finish
    for handle in threads {
        handle.join().unwrap();
    }

    drop(tx); // Drop the sender to signal the printer thread to finish
    printer.join().unwrap(); // Wait for the printer thread to finish

    let collected_entries = Arc::try_unwrap(collected_entries).unwrap().into_inner().unwrap();
    let total_count = Arc::try_unwrap(count).unwrap().into_inner().unwrap();
    (total_count, collected_entries)
}

/// Entry point of the program
fn main() {
    let similarity_threshold = 0.8;
    let path = "/"; // Replace with your target directory's path
    let search_term = "Test"; // The term to compare file names against

    // Start measuring execution time
    let start_time = Instant::now();

    // Exact matches search
    println!("Starting search for exact matches...");
    let exact_start_time = Instant::now();
    let (total_entries_exact, entries) = find_exact_matches_parallel_and_collect(path, search_term);
    let elapsed_time_after_exact = exact_start_time.elapsed();

    println!(
        "Time taken to find all exact matches: {:.2?}",
        elapsed_time_after_exact
    );

    // Similar matches search
    println!("\nStarting search for similar matches...");
    let total_entries_similar = find_similar_matches_parallel_from_vec(
        entries,
        search_term,
        similarity_threshold,
    );

    // Final statistics
    let total_elapsed_time = start_time.elapsed();

    println!(
        "\nTotal files and directories searched: {}",
        total_entries_exact + total_entries_similar
    );
    println!("Total time taken: {:.2?}", total_elapsed_time);
    println!("Finished!");
}

/// Finds similar matches for the search term based on Levenshtein distance (using the prebuilt Vec).
///
/// Returns the number of files processed for similarity.
fn find_similar_matches_parallel_from_vec(
    entries: Vec<walkdir::DirEntry>,
    search_term: &str,
    similarity_threshold: f64,
) -> u32 {
    let (tx, rx) = channel();

    let printer = thread::spawn(move || {
        for message in rx {
            println!("Found similar match: {}", message);
        }
    });

    let count = entries.len() as u32; // Total number of entries processed

    entries.into_iter().for_each(|entry| {
        let file_name = entry.file_name().to_string_lossy();
        let similarity = normalized_levenshtein(&file_name, search_term);
        if similarity >= similarity_threshold {
            let full_path = entry.path().to_string_lossy().into_owned();
            tx.send(full_path).unwrap();
        }
    });

    drop(tx); // Close the channel
    printer.join().unwrap();
    count
}