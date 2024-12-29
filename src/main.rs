use walkdir::WalkDir;
use std::time::Instant;

/// Entry point of the program
fn main() {
    let path = "/home/magnus/"; // Replace with your target directory's path
    let search_term = "";

    // Start measuring time
    let start_time = Instant::now();

    // Call the walk function to get all matches and stats
    let (matches, total_entries) = walk(path, search_term);

    // Stop measuring time
    let elapsed_time = start_time.elapsed();

    // Print the results
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


fn walk(path: &str, search_term: &str) -> (Vec<String>, u32) {
    let mut matches = Vec::new();
    let mut count = 0;

    for entry in WalkDir::new(path) {
        match entry {
            Ok(entry) => {
                count += 1; // Count this entry
                let file_name = entry.file_name().to_string_lossy();
                // Check if the file/directory name contains the search term
                if file_name.contains(search_term) {
                    matches.push(entry.path().to_string_lossy().into_owned());
                }
            }
            Err(e) => println!("Error traversing directory: {}", e),
        }
    }

    (matches, count) // Return the list of matches and the total count
}