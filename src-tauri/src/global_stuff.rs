use std::collections::{HashMap, HashSet};
use tauri::command;

#[command]
pub fn get_fav_extensions() -> Result<HashMap<String, String>, String> {
    let mut extensions: HashMap<String, String> = HashMap::new();

    extensions.insert(String::from("Images"), String::from("png,jpg,jpeg,gif"));
    extensions.insert(String::from("Text"), String::from("txt,doc,docx,pdf,odt,rtf"));
    extensions.insert(String::from("Video"), String::from("mp4,mp4a,avi"));
    extensions.insert(String::from("Coding"), String::from("c,cpp,cs,java,js,html,css,php,py,rs,sh,swift,ts,xml"));

    Ok(extensions)
}