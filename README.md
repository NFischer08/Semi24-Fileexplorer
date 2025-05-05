# A File Explorer with fast and efficient search

short description

## Features

- Fast and efficient file system navigation
- Multi-threaded search algorithm
- Vector space-based file indexing
- SQLite database integration for search optimization
- Cross-platform support (Windows, macOS, Linux)
- Modern and responsive user interface

## Installation

If your system isn't supported, you have to build the application from source code.

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

## Project configuration
There are two config files (`~/data/config/..`):
1. config.json
2. color-config.json

## Contributing

Since this is a personal school project, we are not interested in any contributions.

## Contributors
- [Nino Fischer](https://github.com/NFischer08)
- [Jessica Nolle](https://github.com/Haloooo212)
- [Magnus Schultheis](https://github.com/magnus-52)

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
