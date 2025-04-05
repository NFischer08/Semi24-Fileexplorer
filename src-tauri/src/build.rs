use embed_manifest::embed_manifest_file;

fn main() {
    if cfg!(windows) {
        embed_manifest_file("build.exe.manifest").expect("unable to embed manifest file");
        println!("cargo:rerun-if-changed=sample.exe.manifest");
    }
}
