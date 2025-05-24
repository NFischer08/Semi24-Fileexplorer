use std::path::Path;
use tauri::AppHandle;
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons};
use crate::manager::APP_STATE;

pub fn file_missing_dialog(path_to_file: &Path) {
    let app_handle: AppHandle = APP_STATE.get().expect("I hate life").handle.clone();
    let app_handle_for_closure: AppHandle = app_handle.clone();

    let message = format!(
        "The {} file couldn't be found. Please add the correct path to the config file.",
        path_to_file
            .to_str()
            .expect("file missing dialog failed converting Path to str")
    );

    app_handle
        .dialog()
        .message(message)
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Continue without full functionality".to_string(),
            "Exit Program an Fix Config".to_string(),
        ))
        .show(move |user_clicked_yes| {
            if !user_clicked_yes {
                app_handle_for_closure.exit(0);
            }
        });
}
