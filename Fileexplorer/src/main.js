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
    // Dateien und Ordner abrufen
    const entries = await invoke('format_file_data', { path: filepath });

    // Ergebnisse durchlaufen und in die Tabelle einfügen
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


document.getElementById('file-path-selector').addEventListener('click', async () => {
  await loadFilesAndFolders();
});

document.addEventListener('DOMContentLoaded', async () => {
  await loadFilesAndFolders();
});


window.addEventListener("DOMContentLoaded", () => {
  filePathInputEl = document.querySelector("#file-path");
  resultText = document.querySelector("#result-files");
  document.querySelector("#file-form").addEventListener("submit", (e) => {
    e.preventDefault();
    list_files_in_directory();
  });
});