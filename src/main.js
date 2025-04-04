const { invoke } = window.__TAURI__.core;

let filePathHistory = ["/"];

async function loadFilesAndFolders() {
  let filepath = document.getElementById('file-path-input').value; // get current filepath
  if (filePathHistory[filePathHistory.length] !== filepath) {
    filePathHistory.push(filepath);  // Pfad in die History eintragen
    console.log(filePathHistory);
  }

  const errorMessageElement = document.getElementById('error-message'); // get the errorMessageElement
  errorMessageElement.classList.add('hidden'); // hide error in case there was one

  try {
    const entries = await invoke('format_file_data', { path: filepath }); // grab entries from backend

    const fileTable = document.getElementById('fileTable'); // get the whole table
    fileTable.rows[0].cells[1].style.display = 'none'; // hide column "Filepath"
    fileTable.classList.remove('search') // remove token, so the style gets adjusted properly

    const fileListElement = document.getElementById('fileList'); // get table body with results
    fileListElement.innerHTML = ''; // remove previously displayed files and folders

    entries.forEach(entry => {
      // create new row
      const row = document.createElement('tr');

      // add important background information
      row.dataset.filepath = entry.path; // needed to know the path of the file
      row.dataset.filetype = entry.file_type; // needed to open a file properly when clicking on it

      // create each cell
      const filenameCell = document.createElement('td');
      const lastModifiedCell = document.createElement('td');
      const fileTypeCell = document.createElement('td');
      const fileSizeCell = document.createElement('td');

      // fill each cell with its belonging data
      filenameCell.textContent = entry.name; // file name
      lastModifiedCell.textContent = entry.last_modified; // last time it was modified
      fileTypeCell.textContent = entry.file_type; // file type
      fileSizeCell.textContent = entry.size; //file size

      // add cells to the row
      row.appendChild(filenameCell);
      row.appendChild(lastModifiedCell);
      row.appendChild(fileTypeCell);
      row.appendChild(fileSizeCell);

      // add row to the file list
      fileListElement.appendChild(row);
    });

  } catch (error) {
    console.error('Error:', error);
    // in case of an error it will be displayed beneth the table
    errorMessageElement.textContent = 'Error: ' + error; // set error message
    errorMessageElement.classList.remove('hidden'); // display it
  }
}

async function display_search_results() {
  const search_term = document.getElementById('search-term-input').value; // get the search term
  const search_path = document.getElementById('file-path-input').value; // get the current path

  // get all checked extensions and join them
  const selectedSettings = [];
  for (const checkbox of document.querySelectorAll('input[type="checkbox"]:checked')) {
    selectedSettings.push(checkbox.value); // add the value of the checked checkboxes to the array
  }
  selectedSettings.push(document.getElementById('setting-filetype').value);
  const filetypes = selectedSettings.join(','); // Join the selected values into a string

  const errorMessageElement = document.getElementById('error-message'); // get errorMessageElement
  errorMessageElement.classList.add('hidden'); // remove Error message if it was displayed

  try {
    const entries = await invoke('manager_basic_search', { searchterm: search_term, searchpath: search_path, searchfiletype: filetypes }); // get the search results (structs with all the information)

    const fileListElement = document.getElementById('fileList'); // get table body with results
    fileListElement.innerHTML = ''; // delete previous results

    // load appropriate design and display Filepath column
    const fileTable = document.getElementById('fileTable');
    fileTable.rows[0].cells[1].style.display = ''; // display File Path
    fileTable.classList.add('search'); // design

    // display every result (already sorted by importance)
    entries.forEach(entry => {
      // create new row
      const row = document.createElement('tr');

      // add important background information
      row.dataset.filepath = entry.path
      row.dataset.filetype = entry.file_type

      // create new cells
      const filenameCell = document.createElement('td');
      const filePathCell = document.createElement('td');
      const lastModifiedCell = document.createElement('td');
      const fileTypeCell = document.createElement('td');
      const fileSizeCell = document.createElement('td');

      // insert the information in the cells
      filenameCell.textContent = entry.name; // filename
      filePathCell.textContent = entry.path // filepath
      lastModifiedCell.textContent = entry.last_modified; // last time modified
      fileTypeCell.textContent = entry.file_type; // type
      fileSizeCell.textContent = entry.size; // size

      // append cells to row
      row.appendChild(filenameCell);
      row.appendChild(filePathCell);
      row.appendChild(lastModifiedCell);
      row.appendChild(fileTypeCell);
      row.appendChild(fileSizeCell);

      // insert row into table
      fileListElement.appendChild(row);
    });

  } catch (error) {
    console.error('Error:', error);

    // display error beneth the table
    errorMessageElement.textContent = 'Error: ' + error; // set message
    errorMessageElement.classList.remove('hidden'); // display message
  }
}

// runs when program starts
document.addEventListener('DOMContentLoaded', async () => {
  await display_fav_settings();
  await loadFilesAndFolders();
});

// calls apropriate function when clicking on button
document.getElementById('go-to-file-path-button').addEventListener('click', async () => {
  await loadFilesAndFolders();
});

// calls apropriate function when clicking on button
document.getElementById('search-button').addEventListener('click', async () => {
  if (document.getElementById('search-term-input').value.trim()) { // calls search function only when there is a search term
    await display_search_results();
  }
})

// search settings
const settingsModal = document.getElementById('settings-modal');
// display settings when clicking on button
document.getElementById('settings-button').addEventListener('click', () => {
  settingsModal.classList.remove('hidden');
});
// close settings when clicking away from it
document.getElementById('close-modal').addEventListener('click', () => {
  settingsModal.classList.add('hidden');
});
// TODO was macht das?
window.addEventListener('click', (event) => {
  if (event.target === settingsModal) {
    settingsModal.classList.add('hidden');
  }
});

// request favourite settings from backend (=> config.json)
async function display_fav_settings() {
  const settings = await invoke('get_fav_extensions'); // get favourite settings as HashMap<String, String>
  const form = document.getElementById('settings-form');

  for (const [titel, favourites] of Object.entries(settings)) {
    // create new checkbox with assosiated data
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.id = titel;
    checkbox.value = favourites;

    // create a label for the user with TextNode
    const label = document.createElement('label');
    label.htmlFor = titel;
    label.appendChild(document.createTextNode(titel));

    // append append checkbox and label as well as new line
    form.appendChild(checkbox);
    form.appendChild(label);
    form.appendChild(document.createElement('br'));
  }
}

// double click to open file / folder
document.getElementById('fileList').addEventListener('dblclick', async (event) => {
  event.preventDefault();
  const target = event.target.closest('tr');
  if (target) {
    // open file with program or call function to display folder content
    if (target.dataset.filetype === "Directory") {
      document.getElementById('file-path-input').value = target.dataset.filepath;
      await loadFilesAndFolders();
    } else {
      try {
        invoke('open_file', { filepath: target.dataset.filepath });
      } catch (error) {
        console.error('Error while opening file for user:', error);
      }
    }
  }
})

// submit Filepath by pressing enter
document.getElementById('file-path-input').addEventListener('keypress', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    document.getElementById('go-to-file-path-button').click();
  }
})

// submit search term by pressing enter
document.getElementById('search-term-input').addEventListener('keypress', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    document.getElementById('search-button').click();
  }
})

// backwards buttons
document.getElementById('back-button').addEventListener('click', async () => {
  try {
    const len = filePathHistory.length;
    // check for lenth of 2 or more, because it can't go back if its already in root folder
    if (len < 2) {
      return;
    }
    document.getElementById('file-path-input').value = filePathHistory[len - 2]; // -2 because -1 is current folder => -2 is previous
    filePathHistory.pop(); // remove current path
    filePathHistory.pop(); // remove the one before since it will be added manually
    await loadFilesAndFolders();
  } catch (error) {} // no need to handle error since it just prevents user from going back
});

// context Menu
const contextMenu = document.getElementById('context-menu');
let selectedFile = null;

document.getElementById('fileTable').addEventListener('contextmenu', (event) => {
  event.preventDefault();
  const target = event.target.closest('tr');
  if (target) {
    selectedFile = target.dataset.filepath; // Correctly access the data-file attribute
    contextMenu.style.display = 'block';
    contextMenu.style.left = `${event.pageX}px`;
    contextMenu.style.top = `${event.pageY}px`;
  }
});

// hide context menu as soon as you click on an option
document.addEventListener('click', () => {
  contextMenu.style.display = 'none';
});

// Add click event listeners for context menu options
document.getElementById('context-delete').addEventListener('click', () => {
  if (selectedFile) {
    console.log(`Deleting file: ${selectedFile}`);
    const result = invoke('delete_file', { filepath: selectedFile});
    console.log(result);
    contextMenu.style.display = 'none'; // Hide the menu after action
  }
});

document.getElementById('context-copy').addEventListener('click', () => {
  if (selectedFile) {
    console.log(`Copying file: ${selectedFile}`);
    const result = invoke('copy_file', { filepath: selectedFile});
    console.log(result);
    contextMenu.style.display = 'none'; // Hide the menu after action
  }
});

document.getElementById('context-paste').addEventListener('click', () => {
  if (selectedFile) {
    //let selectedFile = document.getElementById("file-path").value
    console.log(`Pasting file: ${selectedFile}`);
    const result = invoke('paste', { destination: selectedFile});
    console.log(result);
    contextMenu.style.display = 'none'; // Hide the menu after action
  }
});

document.getElementById('context-cut').addEventListener('click', () => {
  if (selectedFile) {
    console.log(`Cutting file: ${selectedFile}`);
    const result = invoke('cut_file', { filepath: selectedFile});
    console.log(result);
    contextMenu.style.display = 'none'; // Hide the menu after action
  }
});

document.getElementById('context-open_with').addEventListener('click', () => {
  if (selectedFile) {
    console.log(`Opening file: ${selectedFile} with`);
    const result = invoke('open_file_with', { filepath: selectedFile});
    console.log(result);
    contextMenu.style.display = 'none'; // Hide the menu after action
  }
});

document.getElementById('context-rename').addEventListener('click', () => {
  if (selectedFile) {
    contextMenu.style.display = 'none'; // Hide the menu after action
  }
});

// TODO: I don't understand it!
document.addEventListener("DOMContentLoaded", () => {
  const renameTrigger = document.getElementById("context-rename");
  const renameModal = document.getElementById("rename-modal");
  const closeRenameModal = document.getElementById("close-rename-modal");
  const renameForm = document.getElementById("rename-form");
  const newFilenameInput = document.getElementById("new-filename");

  // Öffne das Modal bei Klick auf "Rename"
  renameTrigger.addEventListener("click", () => {
    renameModal.classList.remove("hidden");
    newFilenameInput.value = ""; // Texteingabe zurücksetzen
    newFilenameInput.focus(); // Fokussiert die Eingabe
  });

  // Schließen des Modals
  closeRenameModal.addEventListener("click", () => {
    renameModal.classList.add("hidden");
  });

  // Umbenennen bei Abschicken des Formulars
  renameForm.addEventListener("submit",async (e) => {
    e.preventDefault();
    const newFilename = newFilenameInput.value.trim();
    if (newFilename || !newFilename.contains("/")) { // make sure its a valid new name - neither / nor empty
      try {
        const result = invoke('rename_file', {filepath: selectedFile, newFilename: newFilename});
        console.log(`Renaming to: ${newFilename}`); // Hier erfolgt der Umbenennungsprozess
        console.log(result);
        renameModal.classList.add("hidden");
        await loadFilesAndFolders();
      } catch (error) {
        console.error('Error during renaming:', error); // Fehlerprotokollierung
        alert("An error occurred while renaming the file.");
      }
    } else {
      alert("Please enter a valid filename.");
    }
  });
})
