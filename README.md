# A File Explorer with fast and efficient search

In this project, we have implemented a file explorer with fast and efficient search. The application is written in Rust and uses the Tauri framework. The main featuares are:

## Features

- File system navigation
- Multi-threaded search algorithm
- search options to search for specific file types
- Vector space-based file indexing
- SQLite database integration for search optimization
- Cross-platform support (Windows, macOS, Linux)
- Modern and responsive user interface

## Installation

If there is no release for your system, you have to build the application from source code.

### Option 1: Using the Executable

1. Download the latest release from the Releases page
2. Run the installer
3. Launch the application from your system's application menu or by double-clicking it

### Option 2: Building from Source Code

1. Install Rust:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Install Tauri CLI:
   ```bash
   cargo install tauri-cli
   ```

3. Clone the repository:
   ```bash
   git clone https://github.com/NFischer08/Semi24-Fileexplorer
   cd Semi24-Fileexplorer
   ```
   
4. Run

   |        in development mode         |      in release mode      |
   |:----------------------------------:|:-------------------------:|
   | ``` cargo tauri dev --no-watch ``` | ``` cargo tauri build ``` |

## How to use it
First of all you should start the program. 
It will initialise the needed path structure. 
After that you can configure the program to your preferences (see [config](CONFIG.md)).
Now you can use the program just like you want to.

The following apects are supported by our program:

- **right click** on entries (copy, paste, cut, delete, rename and open)
- **double click** on entries to open them
- **navigation buttons**: up to go to the parent folder, left to go to the previous folder and right to revert your click on the left button
- input field to enter a filepath
- input field to enter your search term
- **search settings** when pressing on the three lines with dots

   In here you can either enter custom extensions you want to query for or click on some favourites (set in config).
   You have to use "dir" for directories and "binary" for files without extensions.
   By clicking on the trash can your search setting inputs will be reset.
- pressing _enter_ will submit if you're currently in an input field

Now have fun using our File Explorer 🎉🎉🎉

## Error Handling
In case you encounter any problems try what the error messages tell you to do.
If that doesn't work or the program crashes when starting it often helps to just restart the program.
Furthermore deleting the [database](src-tauri/data/db) (`~/data/db/..`) or deleting the [config files](src-tauri/data/config) can help too

## Project configuration
There are two config files (`~/data/config/..`) for customising the File Explorer:
1. [config.json](src-tauri/data/config/config.json)
2. [color-config.json](src-tauri/data/config/color-config.json)

Read [here](CONFIG.md) to learn more.

## Contributing

Since this is a personal school project, we are not interested in any contributions.
However, if you have any questions or suggestions, feel free to contact us.
Reporting any issues is also appreciated.

## Contributors
This project was created by:
- [Nino Fischer](https://github.com/NFischer08)
- [Jessica Nolle](https://github.com/Haloooo212)
- [Magnus Schultheis](https://github.com/magnus-52)

... as a _Seminarfach_-project at the [Albert-Schweitzer grammer school Erfurt specializing in computer science, natural science and mathematics](https://web.asgspez.de/).
View our coressponding cientific paper: [scientific paper](Seminarfacharbeit.pdf) (German)

## License

his project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

This means you are free to:
- Share: Copy and redistribute the material in any medium or format
- Adapt: Remix, transform, and build upon the material

Under the following terms:
- Attribution: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- NonCommercial: You may not use the material for commercial purposes without explicit permission from the copyright holders.

For commercial use, please contact the project maintainers.

For more details, see the [LICENSE](LICENSE) file in the repository.
