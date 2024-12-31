# Projekt: Dateiverwaltung mit Tauri
Dieses Projekt implementiert eine einfache Dateiverwaltungs-Anwendung, die mit [Tauri]() entwickelt wurde. Die Anwendung ermöglicht es, Dateien und Ordner eines angegebenen Pfades zu laden, deren Eigenschaften anzuzeigen, und dabei eine reibungslose Benutzererfahrung bereitzustellen.
## Projektübersicht
Die Anwendung basiert auf einer Kombination aus **Rust** und **JavaScript (ES6)**. Dabei wird Tauri genutzt, um eine native Desktop-Anwendung zu erstellen. Informationen über Dateien und Ordner werden mit dem Tauri-Kommando-System verwaltet, während eine interaktive Benutzeroberfläche mit HTML, CSS und JavaScript umgesetzt wurde.
Die Hauptfunktionen umfassen:
- Abrufen und Anzeigen von Dateien und Ordnern eines angegebenen Dateipfads.
- Ausgabe von Dateiattributen:
    - **Dateiname**
    - **Letztes Änderungsdatum**
    - **Dateityp**
    - **Dateigröße**

- Fehlerbehandlung und visuelle Rückmeldung bei fehlerhaften Aktionen.

## To-Do Liste
- Search Algorithm:
    - **Threadpool**
    - **Datenbank**
    - **Vectorspaces**

## Projektstruktur
Das Projekt wird in zwei Hauptteilen untergliedert:
### 1. **Frontend** (HTML, CSS, JavaScript)
Hier wird die Benutzeroberfläche definiert. Es gibt Eingabefelder für die Dateipfadeingabe, Buttons zur Navigation und ein Tabellenformat zur Anzeige von Datei- und Ordnerinformationen.
### 2. **Backend** (Tauri und Rust)
Das Backend verarbeitet Anfragen, ruft Dateiinformationen ab und stellt sie dem Frontend zur Verfügung. Mit der `invoke`-Funktion von Tauri wird die Brücke zwischen JavaScript und Rust geschlagen.
## Voraussetzungen
Um das Projekt zu bauen und auszuführen, stellen Sie sicher, dass die folgenden Anforderungen vorliegen:
- [Rust]()
- [Node.js]()
- Tauri CLI:
``` bash
  cargo install tauri-cli
```
## Installation und Ausführung
1. **Repository klonen**:
``` bash
   git clone <github.com/Paulonus28/Semi24-Fileexplorer/>
   cd <PROJEKTORDNER>
```
1. **Abhängigkeiten installieren**: Führen Sie den folgenden Befehl aus, um alle notwendigen Pakete zu installieren:
``` bash
   npm install
```
1. **Entwicklungssitzung starten**: Um die Anwendung in der Entwicklungsumgebung auszuführen, nutzen Sie:
``` bash
   npm run tauri dev
```
1. **Produktion bauen**: Um ein ausführbares Paket zu erstellen, führen Sie Folgendes aus:
``` bash
   npm run tauri build
```
## Nutzung
1. Starten Sie die Anwendung.
2. Geben Sie den gewünschten Verzeichnispfad in das Eingabefeld ein.
3. Klicken Sie auf den Button, um Dateien und Ordner zu laden.
4. Die Anwendung zeigt eine Liste mit den folgenden Informationen an:
    - **Dateiname**
    - **Letzte Änderung**
    - **Dateityp**
    - **Dateigröße**

5. Bei Fehlern, z. B. ungültigen Pfaden, wird eine Fehlermeldung angezeigt.

### Beispiel: Dateianzeige
Falls der eingegebene Pfad gültig ist, zeigt die Anwendung die Dateien und Ordner in einer Tabelle an. Jede Zeile enthält Spalten mit den entsprechenden Datei-Attributen.
### Fehlerbehandlung
Falls ein ungültiger Pfad oder ein anderer Fehler auftritt, wird eine benutzerfreundliche Fehlermeldung unterhalb der Tabelle angezeigt.
## Technologien
In diesem Projekt kommen folgende Technologien und Bibliotheken zum Einsatz:
- **Rust**: Zur Erstellung des Backends.
- **Tauri**: Für plattformübergreifende Desktop-Integration.
- **JavaScript (ES6)**: Zur Steuerung und Manipulation des Frontends.
- **HTML & CSS**: Für die Benutzeroberfläche.
- **Chrono**: Für die Handhabung von Datumsformaten im Backend.
- **serde** und **serde_json**: Für die JSON-Serialisierung und -Deserialisierung.

## Verwendete Rust-Kisten (Dependencies)
Die wichtigsten Rust-Bibliotheken in diesem Projekt:
- `serde` (Version: 1.0.217): Serialisierung und Deserialisierung von Daten.
- `serde_json` (Version: 1.0.134): Umgang mit JSON-Dateien.
- `chrono` (Version: 0.4.39): Handhabung von Zeit- und Datumsformaten.
- `tauri` (Version: 2.1.1): Hauptframework.
- `tauri-plugin-shell`: Zur Unterstützung von Shell-Befehlen im Tauri-Projekt.

## Beiträge
Beiträge, Fehlerberichte und Verbesserungsvorschläge sind willkommen! Reichen Sie diese über die **Issues-Sektion** in diesem Repository ein.
## Lizenz
Dieses Projekt steht unter der **MIT-Lizenz**. Weitere Informationen finden Sie in der im Repository enthaltenen `LICENSE`-Datei.

Vielen Dank, dass Sie dieses Projekt verwenden! 😊
