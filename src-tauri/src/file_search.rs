use threadpool::ThreadPool;
use std::{sync::mpsc::channel, sync::{Arc, Mutex}};
use strsim::normalized_levenshtein;
use walkdir::{DirEntry, WalkDir};

/// Finds exact matches for the search term while building a Vec of entries for further processing.
///
/// Returns a tuple containing the total entries checked during the exact search and the Vec of entries.
fn find_exact_matches_parallel_and_collect(
    path: &str,
    search_term: &str,
    n_workers: usize,
) -> (Vec<String>, Vec<DirEntry>) {
    let collected_entries = Arc::new(Mutex::new(Vec::new())); // Shared collection for entries

    let pool = ThreadPool::new(n_workers);
    let (tx, rx) = channel(); // Channel for communication between threads

    for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
        let tx = tx.clone();
        let collected_entries_clone = Arc::clone(&collected_entries);
        let search_term = search_term.to_owned();
        let entry_clone = entry.clone(); // Clone the entry for use in the thread

        pool.execute(move || {

            let file_name = entry_clone.file_name().to_str().unwrap_or_default().to_owned();

            // Save the entry for later similarity search
            collected_entries_clone.lock().unwrap().push(entry_clone.clone());

            // Check if the entry matches the search term
            if file_name == search_term {
                let full_path = entry_clone.path().to_string_lossy().into_owned();
                tx.send(full_path).unwrap(); // Send the result to the main thread
            }
        });
    }

    // Drop the sender to close the channel
    drop(tx);

    // Collect results from the channel
    let mut results = Vec::new();
    for received in rx {
        println!("{:?}", received); // Print for debugging prints exact matches as Strings sent by tx
        results.push(received);
    }

    // Return the count of entries checked and the collected entries
    let exact_matches = results;
    let collected_entries = collected_entries.lock().unwrap();
    let x = (exact_matches, collected_entries.clone());
    x // Return the total count of entries searched
}

/// Entry point of the program
fn main() {
    let similarity_threshold = 0.8;
    let path = "/"; // Replace with your target directory's path
    let search_term = "test"; // The term to compare file names against

    // Number of Threads used in Exact_Matches and Similar_Matches
    let n_workers = 10;

    // Exact matches search
    println!("Starting search for exact matches...");
    let (exact_matches, entries) = find_exact_matches_parallel_and_collect(path, search_term, n_workers);
    let count = entries.len();
    //exact matches are all found direct matches in a Vec<String>

    // Similar matches search
    println!("\nStarting search for similar matches...");
    let matches_similar = find_similar_matches_parallel_from_vec(
        entries,
        search_term,
        similarity_threshold,
        n_workers,
    );

    println!("\nFinished search");
    println!("{:?}", count)

// matches_similar are all found similar directorys in a Vec<String>
}
/// Finds similar matches for the search term from a vector of directory entries.
///
/// Returns the count of similar matches found.

fn find_similar_matches_parallel_from_vec(
    entries: Vec<DirEntry>,
    search_term: &str,
    similarity_threshold: f64,
    n_workers: usize,
) -> Vec<String> {
    let (tx, rx) = channel();
    let pool = ThreadPool::new(n_workers);

    // Process each entry in parallel
    for entry in entries {
        let tx = tx.clone();
        let search_term = search_term.to_owned();
        let similarity_threshold = similarity_threshold;


        pool.execute(move || {
            let file_name = entry.file_name().to_string_lossy().clone();
            let similarity = normalized_levenshtein(&file_name, &search_term);

            if similarity >= similarity_threshold {
                let full_path = entry.path().to_string_lossy().into_owned();
                tx.send(full_path).unwrap(); // Send the result to the main thread
            }
        });
    }

    drop(tx);

    let mut results = Vec::new();
    for received in rx {
        println!("{:?}", received);
        results.push(received);
    }
    let findings = results;
    findings
}