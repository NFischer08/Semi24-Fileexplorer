use rayon::prelude::*;
use std::{
    sync::mpsc::{channel},
    thread,
    time::Instant,
};
use strsim::normalized_levenshtein;
use walkdir::WalkDir;

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

    // Similar matches search (if exact matches were found or regardless of matches)
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
}

/// Finds exact matches for the search term while building a Vec of entries for further processing.
///
/// Returns a tuple containing the total entries checked during the exact search and the Vec of entries.
fn find_exact_matches_parallel_and_collect(
    path: &str,
    search_term: &str,
) -> (u32, Vec<walkdir::DirEntry>) {
    let start_time = Instant::now(); // Start measuring time
    let mut count = 0; // Counter for the total entries checked
    let mut collected_entries = Vec::new(); // Vector to hold all entries for later use

    // Walk through the filesystem using WalkDir
    for entry in WalkDir::new(path).into_iter().filter_map(|entry| entry.ok()) {
        count += 1; // Increment the counter for every entry processed

        // Save the entry into the vector for later use
        collected_entries.push(entry.clone());

        let file_name = entry.file_name().to_string_lossy();

        // Check for exact match
        if file_name == search_term {
            let full_path = entry.path().to_string_lossy().into_owned();
            println!("Found exact match: {}", full_path);

            // Print the elapsed time since the start of the function
            let elapsed_time = start_time.elapsed();
            println!("Time elapsed: {:.2?}", elapsed_time);
        }
    }

    (count, collected_entries) // Return the total count and the collected entries
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

    // Spawn a thread to receive and print results as they are found
    let printer = thread::spawn(move || {
        for message in rx {
            println!("Found similar match: {}", message);
        }
    });

    let count = entries.len() as u32; // Total number of entries processed

    // Clone `tx` so it can be moved into the Rayon closure while keeping the original
    let tx_clone = tx.clone();
    entries.into_par_iter().for_each_with(tx_clone, |tx, entry| {
        let file_name = entry.file_name().to_string_lossy();

        // Check for similarity
        let similarity = normalized_levenshtein(&file_name, search_term);
        if similarity >= similarity_threshold {
            let full_path = entry.path().to_string_lossy().into_owned();
            tx.send(full_path).unwrap(); // Send the result to the printer thread
        }
    });

    // Drop the sender here to signal the printer thread that no more data is coming
    drop(tx);

    printer.join().unwrap(); // Wait for the printer thread to finish

    count
}


///