# SWG Building IFF Editor (MVP)

This project is a starter desktop-style web tool for editing SWG building definitions.

Current capabilities:

- Load building IFF files
- Load object IFF files
- Load appearance assets
- Manage an in-memory building composition (add/remove objects)
- Edit transform values for placed objects
- Live 2D footprint preview with object markers
- Export your current session as JSON

This is the first milestone. It does not yet write binary SWG IFF output.

## Run

If your PowerShell blocks npm scripts, use npm via node directly:

```powershell
node "C:\Program Files\nodejs\node_modules\npm\bin\npm-cli.js" install
node "C:\Program Files\nodejs\node_modules\npm\bin\npm-cli.js" run dev
```

If your shell allows npm directly:

```powershell
npm install
npm run dev
```

## Main workflow

1. Load at least one building IFF.
2. Load object IFF files.
3. Optionally load appearance files.
4. Select a building and click Add Object.
5. Select placed objects in Inspector and adjust X, Y, Z, YAW, and SCALE.
6. Export Session to save a JSON snapshot.

## Architecture notes

- UI and editor state are currently managed in [src/App.tsx](src/App.tsx).
- Styling is in [src/App.css](src/App.css) and [src/index.css](src/index.css).
- The preview is a canvas renderer intended as a placeholder for a real SWG model renderer.

## Next implementation steps

1. Add a binary IFF parser for building/object chunks.
2. Add a binary writer so edits can be saved back to IFF.
3. Replace the 2D preview with true 3D rendering and appearance mesh loading.
4. Add validation rules for SWG object placement constraints.
5. Add undo/redo and multi-building project tabs.
