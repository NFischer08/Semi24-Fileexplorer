# Config

Our project has a config file with which you can change important parameters for your preferences. 
Furthermore, there is a color config file for changing the appereance of the program.

Note that you have to restart the program for changes to take effect.

## General Config
You can find it [here](src-tauri/data/config/config.json).

### `file_extensions_to_index` 
A list containing all file extensions that should be indexed (added to database) of type string. 
All filetypes not found in this list will be ignored and can therefore not be searched for.

### `favourite_extensions`
An dictionary where you can add your favourite file extensions. 
The key is the label and the value contains the extensions sperated by a comma.
Both the key and the value are of type string.
The label will be displayed in the file explorer search settings as a checkbox.

### `copy_mode`
The copy mode determines how files are copied. 
Since it is of type CopyMode (custom Enum), you can choose between:
1. `Clipboard`: The file contents will be copied to the clipboard. 
This is recommended for small files and if you want to paste the file contents into another program.
2. `File`: The file will be copied to `src-tauri/data/tmp/CONTENT` and the filename will be stored in `src-tauri/data/tmp/copy.txt`.
This is recommended for large files.

Note that you have to give them as a string.

### `number_results_levenshtein`
An integer determining the number of results from the Levenshtein-Distance that should be displayed.

### `number_results_embedding`
An integer determining the number of results from the embedding similarity (from our "Skip-Gram-Model") that should be displayed.

### `paths_to_index`
A list containing all paths that should be indexed (added to database) of type string.
All children of the paths will be indexed as well.

### `index_hidden_files`
A boolean determining if hidden files should be indexed.

### `create_batch_size`
An integer determining the number of files that should be added to the database at once.

### `search_batch_size`
An integer determining the number of files that should be processed for at once while searching.

### `number_of_threads`
An integer determining the number of threads that should be used for processing. 
By entering somthing non valid (like `None`), the default value will be used.
That value is automatically adjusted to one below the number of logical cores of your computer.

### `paths_to_ignore`
A list containing all paths that should be ignored of type string.
This is usefull if you want to ignore specific folders that normally would be indexed.

### `path_to_weights`
A string containing the path to the weights file.
Incase you leave the weights file where it is, you can leave this field empty or write `None`, because it will just use the [default path](src-tauri/data/model).

### `path_to_vocab`
A string containing the path to the vocab file.
Incase you leave the vocab file where it is, you can leave this field empty or write `None`, because it will just use the [default path](src-tauri/data/model).

### `embedding_dimensions`
An integer determining the dimension of the embedding model.
It represents the amount of floats each Entry in the vocab file is embedded into.

## Color Config

You can find it [here](src-tauri/data/config/color-config.json).

### `background`
Defines the background color of the program.

### `font`
Defines the font color of the program.

### `table_head_background`
Defines the background color of the table head.

### `table_every_second_row_background`
Defines the background color of every second row.
All other rows will have the same color as the background.

### `table_border`
Defines the border color of the columns in the table.

### `input_border`
Defines the border color of the input fields.

### `button_hover`
Defines the font color of the top buttons when hovered.

### `modal_background`
Defines the background color of all modals including the rightclick menu, the search settings and the rename form.

### `modal_hover`
Defines the background color of buttons in modals when hovered.
