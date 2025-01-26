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

      filenameCell.textContent = entry.name;
      lastModifiedCell.textContent = entry.last_modified;
      fileTypeCell.textContent = entry.file_type;
      fileSizeCell.textContent = entry.size;

      row.appendChild(filenameCell);
      row.appendChild(lastModifiedCell);
      row.appendChild(fileTypeCell);
      row.appendChild(fileSizeCell);
      fileListElement.appendChild(row);
    });

  } catch (error) {
    console.error('Error:', error);

    // Fehlermeldung unter der Tabelle anzeigen
    errorMessageElement.textContent = 'Error: ' + error;
    errorMessageElement.classList.remove('hidden'); // Meldung sichtbar machen
  }
}

/**
 * async function loadSearchResults() {
 *   //Initialisierung
 *   const filepath = document.getElementById('file-path').value; // Aktuellen Pfad auslesen
 *   const fileListElement = document.getElementById('fileList');
 *   const errorMessageElement = document.getElementById('error-message');
 *   fileListElement.innerHTML = ''; // Vorherige Ergebnisse löschen
 *   errorMessageElement.classList.add('hidden');
 *
 *   try {
 *     const entries = await invoke('search_results', { path: filepath });
 *
 *     entries.forEach(entry => {
 *       const row = document.createElement('tr');
 *       row.dataset.filepath = filepath + "/" + entry.name;
 *
 *       const filenameCell = document.createElement('td');
 *       const lastModifiedCell = document.createElement('td');
 *       const fileTypeCell = document.createElement('td');
 *       const fileSizeCell = document.createElement('td');
 *       const pathCell = document.createElement('tr');
 *
 *       filenameCell.textContent = entry.name; // Dateiname
 *       lastModifiedCell.textContent = entry.last_modified; // Letzte Änderung
 *       fileTypeCell.textContent = entry.file_type; // Dateityp
 *       fileSizeCell.textContent = entry.size; //Größe
 *       pathCell.textContent = entry.path; // Path
 *
 *       row.appendChild(filenameCell);
 *       row.appendChild(lastModifiedCell);
 *       row.appendChild(fileTypeCell);
 *       row.appendChild(fileSizeCell);
 *       fileListElement.appendChild(row);
 *       fileListElement.appendChild(pathCell);
 *     });
 *
 *   } catch (error) {
 *     console.error('Error:', error);
 *
 *     // Fehlermeldung unter der Tabelle anzeigen
 *     errorMessageElement.textContent = 'Error: ' + error; // Fehlermeldung setzen
 *     errorMessageElement.classList.remove('hidden'); // Meldung sichtbar machen
 *   }
 * }
 */


document.getElementById('file-path-selector').addEventListener('click', async () => {
  await loadFilesAndFolders();
});

const toggleButton = document.getElementById('toggle-sidebar');
const sidebar = document.querySelector('.sidebar');

toggleButton.addEventListener('click', () => {
  sidebar.classList.toggle('collapsed'); // Toggle der Klasse für das Ein- und Ausklappen
  toggleButton.textContent = sidebar.classList.contains('collapsed') ? '>' : '<'; // Ändern des Pfeils
});

document.addEventListener('DOMContentLoaded', async () => {
  await loadFilesAndFolders();
});

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
    console.log(selectedFile);
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
    // Add your delete logic here
    contextMenu.style.display = 'none'; // Hide the menu after action
  }
});

document.getElementById('context-copy').addEventListener('click', () => {
  if (selectedFile) {
    console.log(`Copying file: ${selectedFile}`);
    // Add your copy logic here
    contextMenu.style.display = 'none'; // Hide the menu after action
  }
});

document.getElementById('context-rename').addEventListener('click', () => {
  if (selectedFile) {
    console.log(`Renaming file: ${selectedFile}`);
    // Add your rename logic here
    contextMenu.style.display = 'none'; // Hide the menu after action
  }
});

document.getElementById('context-cut').addEventListener('click', () => {
  if (selectedFile) {
    console.log(`Cutting file: ${selectedFile}`);
    // Add your cut logic here
    contextMenu.style.display = 'none'; // Hide the menu after action
  }
});

document.getElementById('context-open_with').addEventListener('click', () => {
  if (selectedFile) {
    console.log(`Opening file: ${selectedFile} with`);
    // Add your open_with logic here
    contextMenu.style.display = 'none'; // Hide the menu after action
  }
});

window.addEventListener("DOMContentLoaded", () => {
  filePathInputEl = document.querySelector("#file-path");
  resultText = document.querySelector("#result-files");
  document.querySelector("#file-form").addEventListener("submit", (e) => {
    e.preventDefault();
    list_files_in_directory();
  });
});

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
}