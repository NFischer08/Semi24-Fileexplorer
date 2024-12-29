use walkdir::WalkDir;
use strsim::normalized_levenshtein;
use std::time::Instant;

/// Entry point of the program
fn main() {
    let path = "/home/magnus/"; // Replace with your target directory's path
    let search_term = "Test";   // Term to compare file names against

    // Start measuring execution time
    let start_time = Instant::now();

    // First, check for exact matches
    let (exact_matches, total_entries_exact, elapsed_time_after_exact) =
        find_exact_matches(path, search_term);

    // Then, check for similar matches
    let (similar_matches, total_entries_similar) =
        find_similar_matches(path, search_term, 0.8);

    let total_elapsed_time = start_time.elapsed();

    // Display exact matches
    if exact_matches.is_empty() {
        println!("No exact match found for '{}'", search_term);
    } else {
        println!("Found exact matches:");
        for m in exact_matches {
            println!("{}", m);
        }
        // Print the time after finding all exact matches
        println!(
            "Time taken to find all exact matches: {:.2?}",
            elapsed_time_after_exact
        );
    }

    // Display similar matches
    if !similar_matches.is_empty() {
        println!("\nFound similar matches (based on Levenshtein distance):");
        for m in similar_matches {
            println!("{}", m);
        }
    }

    // Print statistics
    println!("\nTotal files and directories searched: {}", total_entries_exact + total_entries_similar);
    println!("Total time taken: {:.2?}", total_elapsed_time);
}

/// Finds exact matches for the search term in the given directory.
///
/// Returns a tuple of the exact matching file paths,
/// the total entries checked, and the elapsed time.
fn find_exact_matches(path: &str, search_term: &str) -> (Vec<String>, u32, std::time::Duration) {
    let mut exact_matches = Vec::new();
    let mut count = 0; // Total entries processed
    let start_time = Instant::now();

    for entry in WalkDir::new(path) {
        match entry {
            Ok(entry) => {
                count += 1; // Increment the total number of visited entries
                let file_name = entry.file_name().to_string_lossy();

                // Check for exact match
                if file_name == search_term {
                    exact_matches.push(entry.path().to_string_lossy().into_owned());
                }
            }
            Err(e) => println!("Error traversing directory: {}", e),
        }
    }

    // Get elapsed time for finding all exact matches
    let elapsed_time = start_time.elapsed();

    (exact_matches, count, elapsed_time)
}

/// Finds similar matches for the search term based on Levenshtein distance.
///
/// Returns a tuple of the similar matching file paths and the total entries checked.
fn find_similar_matches(path: &str, search_term: &str, similarity_threshold: f64) -> (Vec<String>, u32) {
    let mut similar_matches = Vec::new();
    let mut count = 0; // Total entries processed

    for entry in WalkDir::new(path) {
        match entry {
            Ok(entry) => {
                count += 1; // Increment the total number of visited entries
                let file_name = entry.file_name().to_string_lossy();

                // Check for similarity
                let similarity = normalized_levenshtein(&file_name, search_term);
                if similarity >= similarity_threshold {
                    similar_matches.push(entry.path().to_string_lossy().into_owned());
                }
            }
            Err(e) => println!("Error traversing directory: {}", e),
        }
    }

    (similar_matches, count)
}