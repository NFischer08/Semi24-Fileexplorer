const { invoke } = window.__TAURI__.core;

let filePathInputEl;
let resultText;

/*document.getElementById('file-path-selector').addEventListener('click', async () => {
  const filepath = document.getElementById('file-path').value;
  const fileListElement = document.getElementById('fileList');
  fileListElement.innerHTML = ''; // Clear previous results

  try {
    const entries = await invoke('format_file_data', { path: filepath });
    entries.forEach(entry => {
      const row = document.createElement('tr');
      const filenameCell = document.createElement('td');
      const lastModifiedCell = document.createElement('td');
      const fileTypeCell = document.createElement('td');
      const fileSizeCell = document.createElement('td');

      filenameCell.textContent = entry.name; // Display the filename
      lastModifiedCell.textContent = entry.last_modified; // Display last modified date
      fileTypeCell.textContent = entry.file_type; // Display file type
      fileSizeCell.textContent = entry.size

      row.appendChild(filenameCell);
      row.appendChild(lastModifiedCell);
      row.appendChild(fileTypeCell);
      row.appendChild(fileSizeCell);
      fileListElement.appendChild(row);
    });
  } catch (error) {
    console.error('Error:', error);
    const row = document.createElement('tr');
    const errorCell = document.createElement('td');
    errorCell.colSpan = 3; // Span across all columns
    errorCell.textContent = 'Error: ' + error; // Display the error message
    errorCell.classList.add('error'); // Add error class for styling
    row.appendChild(errorCell);
    fileListElement.appendChild(row);
  }
});*/
async function loadFilesAndFolders() {
  const filepath = document.getElementById('file-path').value; // Aktuellen Pfad auslesen
  const fileListElement = document.getElementById('fileList');
  fileListElement.innerHTML = ''; // Vorherige Ergebnisse löschen

  try {
    // Dateien und Ordner abrufen
    const entries = await invoke('list_files_and_folders', { path: filepath });

    // Ergebnisse durchlaufen und in die Tabelle einfügen
    entries.forEach(entry => {
      const row = document.createElement('tr');
      const filenameCell = document.createElement('td');
      const lastModifiedCell = document.createElement('td');
      const fileTypeCell = document.createElement('td');
      const fileSizeCell = document.createElement('td');

      filenameCell.textContent = entry.name; // Dateiname anzeigen
      lastModifiedCell.textContent = entry.last_modified; // Letzte Änderung anzeigen
      fileTypeCell.textContent = entry.file_type; // Dateityp anzeigen
      fileSizeCell.textContent = entry.size;

      row.appendChild(filenameCell);
      row.appendChild(lastModifiedCell);
      row.appendChild(fileTypeCell);
      row.appendChild(fileSizeCell);
      fileListElement.appendChild(row);
    });
  } catch (error) {
    console.error('Error:', error);
    const row = document.createElement('tr');
    const errorCell = document.createElement('td');
    errorCell.colSpan = 3; // Über alle Spalten spannen
    errorCell.textContent = 'Error: ' + error; // Fehlermeldung anzeigen
    errorCell.classList.add('error'); // Fehlerklasse hinzufügen
    row.appendChild(errorCell);
    fileListElement.appendChild(row);
  }
}


document.getElementById('file-path-selector').addEventListener('click', async () => {
  await loadFilesAndFolders(); // Logik aufrufen
});

document.addEventListener('DOMContentLoaded', async () => {
  await loadFilesAndFolders(); // Aufruf der Logik
});


window.addEventListener("DOMContentLoaded", () => {
  filePathInputEl = document.querySelector("#file-path");
  resultText = document.querySelector("#result-files");
  document.querySelector("#file-form").addEventListener("submit", (e) => {
    e.preventDefault();
    list_files_in_directory();
  });
});