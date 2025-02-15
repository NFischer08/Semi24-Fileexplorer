const { invoke } = window.__TAURI__.core;

let filePathInputEl;
let resultText;

async function loadFilesAndFolders() {
  const filepath = document.getElementById('file-path').value; // Aktuellen Pfad auslesen
  const fileListElement = document.getElementById('fileList');
  const errorMessageElement = document.getElementById('error-message');
  fileListElement.innerHTML = ''; // Vorherige Ergebnisse löschen
  errorMessageElement.classList.add('hidden');

  try {
    const entries = await invoke('format_file_data', { path: filepath });

    entries.forEach(entry => {
      const row = document.createElement('tr');
      row.dataset.filepath = filepath + "/" + entry.name;

      const filenameCell = document.createElement('td');
      const lastModifiedCell = document.createElement('td');
      const fileTypeCell = document.createElement('td');
      const fileSizeCell = document.createElement('td');

      filenameCell.textContent = entry.name; // Dateiname
      lastModifiedCell.textContent = entry.last_modified; // Letzte Änderung
      fileTypeCell.textContent = entry.file_type; // Dateityp
      fileSizeCell.textContent = entry.size; //Größe

      row.appendChild(filenameCell);
      row.appendChild(lastModifiedCell);
      row.appendChild(fileTypeCell);
      row.appendChild(fileSizeCell);
      fileListElement.appendChild(row);
    });

  } catch (error) {
    console.error('Error:', error);

    // Fehlermeldung unter der Tabelle anzeigen
    errorMessageElement.textContent = 'Error: ' + error; // Fehlermeldung setzen
    errorMessageElement.classList.remove('hidden'); // Meldung sichtbar machen
  }
}

async function display_search_results() {
  const search_term = document.getElementById('file-name').value; // Aktuellen Pfad auslesen
  const fileListElement = document.getElementById('fileList');
  const errorMessageElement = document.getElementById('error-message');
  fileListElement.innerHTML = ''; // Vorherige Ergebnisse löschen
  errorMessageElement.classList.add('hidden');

  try {
    const names = await invoke('search_database', { search_term: search_term });

    names.forEach(name => {
      const row = document.createElement('tr');
      row.dataset.filepath = filepath + "/" + entry.name;

      const filenameCell = document.createElement('td');
      const lastModifiedCell = document.createElement('td');
      const fileTypeCell = document.createElement('td');
      const fileSizeCell = document.createElement('td');

      filenameCell.textContent = name; // Dateiname
      lastModifiedCell.textContent = "-TODAY-"; // Letzte Änderung
      fileTypeCell.textContent = "--"; // Dateityp
      fileSizeCell.textContent = "--"; //Größe

      row.appendChild(filenameCell);
      row.appendChild(lastModifiedCell);
      row.appendChild(fileTypeCell);
      row.appendChild(fileSizeCell);
      fileListElement.appendChild(row);
    });

  } catch (error) {
    console.error('Error:', error);

    // Fehlermeldung unter der Tabelle anzeigen
    errorMessageElement.textContent = 'Error: ' + error; // Fehlermeldung setzen
    errorMessageElement.classList.remove('hidden'); // Meldung sichtbar machen
  }
}


document.getElementById('file-path-selector').addEventListener('click', async () => {
  await loadFilesAndFolders();
});

document.addEventListener('DOMContentLoaded', async () => {
  await loadFilesAndFolders();
});

document.getElementById('search-button').addEventListener('click', async () => {
  await display_search_results();
})

//Einstellungskästchen
const settingsButton = document.getElementById('settings-button');
const settingsModal = document.getElementById('settings-modal');
const closeModal = document.getElementById('close-modal');


settingsButton.addEventListener('click', () => {
  settingsModal.classList.remove('hidden');
});

closeModal.addEventListener('click', () => {
  settingsModal.classList.add('hidden');
});

window.addEventListener('click', (event) => {
  if (event.target === settingsModal) {
    settingsModal.classList.add('hidden');
  }
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
    console.log(`Renaming file: ${selectedFile}`);
    //const result = invoke('rename_file', { filepath: selectedFile, newFilename: "TEstoto"});
    //console.log(result);
    contextMenu.style.display = 'none'; // Hide the menu after action
  }
});

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
    if (newFilename) {
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

// double click to open file / folder
document.getElementById('fileTable').addEventListener('dblclick', async (event) => {
  event.preventDefault();
  const target = event.target.closest('tr');
  if (target) {
    selectedFile = target.dataset.filepath
    console.log(`Opening file: ${selectedFile}`);
    document.getElementById('file-path').value = selectedFile;
    await loadFilesAndFolders();
  }
})

// submit Filepath by pressing enter
document.getElementById('file-path').addEventListener('keypress', (event) => {
  if (event.key === 'Enter') {
    event.preventDefault();
    document.getElementById('file-path-selector').click();
  }
})


/*

async function loadSearchResults() {
  //Initialisierungq
  const filepath = document.getElementById('file-path').value; // Aktuellen Pfad auslesen
  const fileListElement = document.getElementById('fileList');
  const errorMessageElement = document.getElementById('error-message');
  fileListElement.innerHTML = ''; // Vorherige Ergebnisse löschen
  errorMessageElement.classList.add('hidden');

  try {
    const entries = await invoke('search_results', { path: filepath });

    entries.forEach(entry => {
      const row = document.createElement('tr');
      row.dataset.filepath = filepath + "/" + entry.name;

      const filenameCell = document.createElement('td');
      const lastModifiedCell = document.createElement('td');
      const fileTypeCell = document.createElement('td');
      const fileSizeCell = document.createElement('td');
      const pathCell = document.createElement('tr');

      filenameCell.textContent = entry.name; // Dateiname
      lastModifiedCell.textContent = entry.last_modified; // Letzte Änderung
      fileTypeCell.textContent = entry.file_type; // Dateityp
      fileSizeCell.textContent = entry.size; //Größe
      pathCell.textContent = entry.path; // Path

      row.appendChild(filenameCell);
      row.appendChild(lastModifiedCell);
      row.appendChild(fileTypeCell);
      row.appendChild(fileSizeCell);
      fileListElement.appendChild(row);
      fileListElement.appendChild(pathCell);
    });

  } catch (error) {
    console.error('Error:', error);

    // Fehlermeldung unter der Tabelle anzeigen
    errorMessageElement.textContent = 'Error: ' + error; // Fehlermeldung setzen
    errorMessageElement.classList.remove('hidden'); // Meldung sichtbar machen
  }
}


document.addEventListener('DOMContentLoaded', () => {
  loadPinnedDirectories();

  document.getElementById('add-pinned').addEventListener('click', () => {
    const dir = prompt('Enter directory path to pin:');
    if (dir) {
      addPinnedDirectory(dir);
    }
  });
});

function loadPinnedDirectories() {
  const pinnedList = document.getElementById('pinned-list');
  const directories = JSON.parse(localStorage.getItem('pinnedDirectories')) || [];

  pinnedList.innerHTML = ''; // Leere Listeneinträge löschen
  directories.forEach((dir, index) => {
    const li = document.createElement('li');
    li.textContent = dir;

    const removeButton = document.createElement('button');
    removeButton.textContent = '✖';
    removeButton.style.marginLeft = '10px';
    removeButton.style.background = 'none';
    removeButton.style.color = '#fff';
    removeButton.style.border = 'none';
    removeButton.style.cursor = 'pointer';

    removeButton.addEventListener('click', () => {
      removePinnedDirectory(index);
    });

    li.appendChild(removeButton);
    pinnedList.appendChild(li);

    // Direkter Jump bei Klick
    li.addEventListener('click', () => {
      document.getElementById('file-path').value = dir;
      loadFilesAndFolders();
    });
  });
}

function addPinnedDirectory(dir) {
  const directories = JSON.parse(localStorage.getItem('pinnedDirectories')) || [];
  directories.push(dir);
  localStorage.setItem('pinnedDirectories', JSON.stringify(directories));
  loadPinnedDirectories();
}

function removePinnedDirectory(index) {
  const directories = JSON.parse(localStorage.getItem('pinnedDirectories')) || [];
  directories.splice(index, 1);
  localStorage.setItem('pinnedDirectories', JSON.stringify(directories));
  loadPinnedDirectories();
} */

window.addEventListener("DOMContentLoaded", () => {
  filePathInputEl = document.querySelector("#file-path");
  resultText = document.querySelector("#result-files");
  document.querySelector("#file-form").addEventListener("submit", (e) => {
    e.preventDefault();
    list_files_in_directory();
  });
});