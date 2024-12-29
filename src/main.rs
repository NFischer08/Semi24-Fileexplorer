use walkdir::WalkDir;
use strsim::normalized_levenshtein;
use std::time::Instant;

/// Entry point of the program
fn main() {
    let path = "/home/magnus/"; // Replace with your target directory's path
    let search_term = "Test";   // Term to compare file names against

    // Measure execution time
    let start_time = Instant::now();

    // Call the walk function to find matches based on similarity
    let (matches, total_entries) = walk(path, search_term);

    let elapsed_time = start_time.elapsed();

    // Display results
    if matches.is_empty() {
        println!("No match found for '{}'", search_term);
    } else {
        println!("Found the following matches:");
        for m in matches {
            println!("{}", m);
        }
    }

    // Print statistics
    println!("Total files and directories searched: {}", total_entries);
    println!("Time taken: {:.2?}", elapsed_time);
}

/// Walks through all the files and directories in a given path,
/// comparing file names based on the Levenshtein distance threshold.
///
/// Returns a tuple of matching file paths and the total number of entries visited.
fn walk(path: &str, search_term: &str) -> (Vec<String>, u32) {
    let mut matches = Vec::new();
    let mut count = 0; // Total entries processed
    let similarity_threshold = 0.8; // Define similarity threshold (0.0 to 1.0)

    for entry in WalkDir::new(path) {
        match entry {
            Ok(entry) => {
                count += 1; // Increment the total number of visited entries

                // Only process file names
                let file_name = entry.file_name().to_string_lossy();

                // Compute Levenshtein similarity between file name and search term
                let similarity = normalized_levenshtein(&file_name, search_term);

                // If similarity is above the threshold, add the path to results
                if similarity >= similarity_threshold {
                    matches.push(entry.path().to_string_lossy().into_owned());
                }
            }
            Err(e) => println!("Error traversing directory: {}", e),
        }
    }

    (matches, count) // Return matching paths and total count
}