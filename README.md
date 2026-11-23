# Inlay

**Extract the foreground image from a screenshot.** Inlay is a local-first command-line utility for detecting a prominent image displayed over an application interface, including blurred comment-viewer backgrounds.

> Early implementation: detection scores are heuristic and have not yet been calibrated against the 200-image validation set described in the product requirements. Treat automatic extraction as experimental until that validation is complete.

## Browser proof of concept

Inlay includes a standalone Blazor WebAssembly app that runs the existing C# detector with OpenCvSharp's WebAssembly runtime. Image decoding, detection, and PNG encoding happen in the browser; screenshots are not uploaded.

Requires the .NET 11 SDK and the WebAssembly build tools workload:

```powershell
dotnet workload install wasm-tools
dotnet run --project src/Inlay.Web
```

For a static production build:

```powershell
dotnet publish src/Inlay.Web -c Release
```

The published files are written to `src/Inlay.Web/bin/Release/net11.0/publish/wwwroot`. They can be hosted as static files, including on GitHub Pages. The deployment must preserve the app's repository subpath and publish this directory; no server-side .NET hosting is required.

The browser app accepts batches of up to 50 images (100 MB total, 25 MB per image) and processes them sequentially. Each image shows its own reading, detection, preview, and completion state; finished crops can be inspected and downloaded while the remaining files continue. It also supports direct crop adjustment, corner-radius control, a live PNG preview, and local download. Detection is still experimental; when it cannot find a candidate, you can define the crop manually.

## Requirements

- .NET 11 SDK (11.0.100 or later feature band)
- Windows x64 for the initial native CLI runtime
- .NET WebAssembly build tools workload for the browser app

## CLI usage

```powershell
inlay screenshot.jpg
inlay .\screenshots --recursive --out .\extracted
inlay screenshot.jpg --preview --report
inlay screenshot.jpg --rect 26,592,869,867
inlay screenshot.jpg --corners keep
inlay screenshot.jpg --overwrite
```

Supported input formats: PNG, JPEG/JPG, and WebP. PNG is used for extracted output so transparent rounded corners can be retained.

## Detection behavior

- **Extracted:** the candidate exceeds the initial automatic threshold and is sufficiently separated from the runner-up.
- **Review:** a plausible candidate exists, but its score or separation is insufficient. No crop is exported.
- **Rejected:** no candidate reaches the minimum threshold. No crop is exported.

The score is a ranking heuristic, not a probability. Thresholds must be calibrated on a labelled, held-out dataset before accuracy claims are made.

## Output

For a single file, the default output is `<name>.inlay.png` beside the source. For a directory, the default output directory is `inlay/` beneath the input directory. With `--recursive`, relative subdirectory structure is retained. Existing files are not replaced unless `--overwrite` is supplied.

`--preview` writes a JPEG diagnostic with the best candidate and (when present) runner-up. `--report` writes a JSON report. Review and rejected cases do not produce extracted image files.

## Development

```powershell
dotnet restore
dotnet build --configuration Release
dotnet test --configuration Release
```

The repository's existing CI/deployment setup is intentionally not modified by this change.

## Privacy

The browser app processes images locally using WebAssembly. The CLI processes images locally as well. Neither sends image data to a server.

## Current limitations

- The first detector is a conservative classical-CV baseline, not a trained or calibrated production detector.
- The browser app processes each batch sequentially rather than detecting multiple images in parallel; each batch is limited to 50 images and 100 MB total, with a 25 MB per-image limit.
- Rounded-corner radius estimation is based on contour geometry and requires further real-world validation.
- Only one dominant foreground image is targeted.
- Windows x64 is the supported native CLI runtime target for this first implementation.
