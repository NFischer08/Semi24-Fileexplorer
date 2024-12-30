fn main() {
    println!("cargo:rerun-if-env-changed=TAURI_CONF");

    // Check if the TAURI_CONF environment variable is set
    match std::env::var("/home/magnus/RustroverProjects/Semi24-Fileexplorer/Fileexplorer/src-tauri/tauri.conf.json") {
        Ok(config_path) => {
            println!("cargo:warning=Using Tauri config at: {}", config_path);
        }
        Err(_) => {
            println!("cargo:warning=No TAURI_CONF environment variable set, using default location.");
        }
    }
}