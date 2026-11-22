# Inlay

**Extract the foreground image from a screenshot.** Inlay is a local-first command-line utility for detecting a prominent image displayed over an application interface, including blurred comment-viewer backgrounds.

> Early implementation: detection scores are heuristic and have not yet been calibrated against the 200-image validation set described in the product requirements. Treat automatic extraction as experimental until that validation is complete.

## Requirements

- .NET 11 SDK (11.0.100 or later feature band)
- Windows x64 for the initial native OpenCV runtime

## Usage

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

The GitHub Actions workflow builds and tests on every push and pull request, publishes a self-contained Windows x64 executable, and uploads build/test outputs as downloadable workflow artifacts.

## Privacy

All image processing happens locally. Inlay does not make network requests, send images elsewhere, or include image contents in reports.

## Current limitations

- The first detector is a conservative classical-CV baseline, not a trained or calibrated production detector.
- EXIF orientation is delegated to OpenCV's standard color-image loading behavior.
- Rounded-corner radius estimation is based on contour geometry and requires further real-world validation.
- Only one dominant foreground image is targeted.
- Windows x64 is the supported runtime target for this first implementation.
