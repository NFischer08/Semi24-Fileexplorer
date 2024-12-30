use embed_manifest::embed_manifest_file;


fn main() {
    println!("cargo:rerun-if-env-changed=TAURI_CONF");

    // Check if the TAURI_CONF environment variable is set
    match std::env::var("tauri.conf.json") {
        Ok(config_path) => {
            println!("cargo:warning=Using Tauri config at: {}", config_path);
        }
        Err(_) => {
            println!("cargo:warning=No TAURI_CONF environment variable set, using default location.");
        }
    }
    embed_manifest_file("build.exe.manifest")
        .expect("unable to embed manifest file");
    println!("cargo:rerun-if-changed=sample.exe.manifest");
}