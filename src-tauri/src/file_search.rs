use threadpool::ThreadPool;
use std::{fs::File, io::Write, sync::mpsc::channel, sync::{Arc, Mutex}, thread, time::Instant};
use strsim::normalized_levenshtein;
use walkdir::WalkDir;

/// Finds exact matches for the search term while building a Vec of entries for further processing.
///
/// Returns a tuple containing the total entries checked during the exact search and the Vec of entries.
fn find_exact_matches_parallel_and_collect(
    path: &str,
    search_term: &str,
    n_workers: usize,
) -> (u32, Vec<walkdir::DirEntry>) {
    let start_time = Instant::now(); // Start measuring time
    let count = Arc::new(Mutex::new(0)); // Shared counter for all entries
    let collected_entries = Arc::new(Mutex::new(Vec::new())); // Shared collection for entries

    let pool = ThreadPool::new(n_workers);
    let (tx, rx) = channel(); // Channel for communication between threads

    for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
        let tx = tx.clone();
        let count_clone = Arc::clone(&count);
        let collected_entries_clone = Arc::clone(&collected_entries);
        let search_term = search_term.to_owned();
        let entry_clone = entry.clone(); // Clone the entry for use in the thread

        pool.execute(move || {
            // Increment the counter for each entry checked
            {
                let mut count_lock = count_clone.lock().unwrap();
                *count_lock += 1; // Increment for every entry processed
            }

            let file_name = entry_clone.file_name().to_str().unwrap_or_default().to_owned();

            // Save the entry for later similarity search
            collected_entries_clone.lock().unwrap().push(entry_clone.clone());

            // Check if the entry matches the search term
            if file_name == search_term {
                let full_path = entry_clone.path().to_string_lossy().into_owned();
                tx.send(full_path).unwrap(); // Send the result to the main thread
                println!("{:?} matches the search term.", entry_clone.file_name());
            }
        });
    }

    // Drop the sender to close the channel
    drop(tx);

    // Collect results from the channel
    let mut results = Vec::new();
    for received in rx {
        results.push(received);
    }

    // Count the number of exact matches
    let exact_match_count = results.len() as u32;

    // Return the count of entries checked and the collected entries
    let collected_entries = collected_entries.lock().unwrap();
    let x = (count.lock().unwrap().clone(), collected_entries.clone()); x // Return the total count of entries searched
}

/// Entry point of the program
fn main() {
    let similarity_threshold = 0.8;
    let path = "/"; // Replace with your target directory's path
    let search_term = "Test"; // The term to compare file names against

    // Start measuring execution time
    let start_time = Instant::now();

    // Number of Threads used in Exact_Matches and Similar_Matches
    let n_workers = 10;

    // Exact matches search
    println!("Starting search for exact matches...");
    let exact_start_time = Instant::now();
    let (total_entries_exact, entries) = find_exact_matches_parallel_and_collect(path, search_term, n_workers);
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
        n_workers,
    );

    // Final statistics
    let total_elapsed_time = start_time.elapsed();
    /*
        println!(
            "\nTotal files and directories searched: {}",
            total_entries_exact + total_entries_similar
        );
        println!("Total time taken: {:.2?}", total_elapsed_time);

        // Write the total time and file count to "time.txt"
        let mut file = File::create("time.txt").expect("Unable to create time.txt");
        writeln!(file, "Total files and directories searched: {}", total_entries_exact + total_entries_similar)
            .expect("Failed to write to time.txt");

     */
}
/// Finds similar matches for the search term from a vector of directory entries.
///
/// Returns the count of similar matches found.

fn find_similar_matches_parallel_from_vec(
    entries: Vec<walkdir::DirEntry>,
    search_term: &str,
    similarity_threshold: f64,
    n_workers: usize,
) {
    let (tx, rx) = channel();
    let pool = ThreadPool::new(n_workers);

    // Spawn a thread to print messages received from the channel
    let printer = thread::spawn(move || {
        for message in rx {
            println!("Found similar match: {}", message);
        }
    });

    let count = Arc::new(Mutex::new(0)); // Shared counter for similar matches

    // Process each entry in parallel
    for entry in entries {
        let tx = tx.clone();
        let search_term = search_term.to_owned();
        let similarity_threshold = similarity_threshold;
        let count_clone = Arc::clone(&count);

        pool.execute(move || {
            let file_name = entry.file_name().to_string_lossy();
            let similarity = normalized_levenshtein(&file_name, &search_term);

            if similarity >= similarity_threshold {
                let full_path = entry.path().to_string_lossy().into_owned();
                tx.send(full_path).unwrap(); // Send the result to the main thread

                // Increment the count of similar matches
                let mut count_lock = count_clone.lock().unwrap();
                *count_lock += 1;

            }
        });
    }
    // Drop the sender to close the channel
    drop(tx);
}