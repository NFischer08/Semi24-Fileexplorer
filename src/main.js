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

async function loadSearchResults() {
  //Initialisierung
  const filepath = document.getElementById('file-path').value; // Aktuellen Pfad auslesen
  const fileListElement = document.getElementById('fileList');
  const errorMessageElement = document.getElementById('error-message');
  fileListElement.innerHTML = ''; // Vorherige Ergebnisse löschen
  errorMessageElement.classList.add('hidden');

  try {
    const entries = await invoke('search_results', { path: filepath });

    entries.forEach(entry => {
      const row = document.createElement('tr');
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

document.getElementById('file-path-selector').addEventListener('click', async () => {
  await loadFilesAndFolders();
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