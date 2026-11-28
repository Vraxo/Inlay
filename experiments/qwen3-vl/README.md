# Inlay — hybrid OpenCV and Qwen foreground-panel experiment

This experiment tests a hybrid approach to locating the sharp foreground photo/image panel inside a screenshot whose surrounding backdrop is blurred. OpenCV proposes rectangles from coherent image edges and contours, Qwen judges the four boundaries independently with a sharpness map, and Python combines the selected edges before the final crop and local refinement.

It is deliberately separate from Inlay's .NET detector and the SAM 3 experiment. It does not install PyTorch or download model weights.

## Pipeline

1. Start the LM Studio local API server if it is not already running.
2. Ensure the selected vision-language model is available. If it is not loaded, Inlay can discover and load its local model key.
3. Load the original screenshot with Pillow and correct its EXIF orientation.
4. OpenCV computes horizontal and vertical Sobel gradients, builds long-edge projection profiles, and finds large closed-contour rectangles.
5. The detector combines the strongest coherent edge pair with other high-scoring geometric candidates. Candidate scoring considers four-edge support, projection strength, area, and centrality.
6. Python measures local Laplacian detail energy immediately inside and outside every proposed edge. A valid photo boundary should have more sharp detail on its inside and less detail on its outside. It ranks the left, top, right, and bottom coordinates independently, using the shared interior of the proposals as the sampling region.
7. If all four edge choices have strong inside-versus-outside contrast, sufficient detail inside, and a clear lead over alternatives, Python combines them directly. This avoids a model call when the sharp/blurred transition is unambiguous. The initial conservative gates require a combined contrast score >= 0.08, inside detail >= 10% of the image reference level, and a margin >= 0.035 over the next edge position.
8. Otherwise, Python creates three diagnostic images: numbered candidate rectangles, a sharpness map, and a four-panel boundary atlas. The map uses local Laplacian detail energy; brighter areas are sharper and darker areas are blurrier. Each atlas panel shows only one edge direction and labels the possible positions with candidate IDs and pixel coordinates. The original, candidate overview, sharpness map, and boundary atlas are sent to Qwen through LM Studio's native `/api/v1/chat` endpoint.
9. Qwen selects the left, top, right, and bottom boundaries independently, using the matching panel in the atlas. Each response field references a candidate ID whose corresponding edge coordinate should be used. The four IDs may differ, allowing Python to combine boundaries from different proposals. Qwen does not generate or estimate coordinates.
10. If sharpness evidence is inconclusive, the existing conservative four-edge OpenCV rectangle gate remains available before Qwen. Python validates the combined rectangle, maps it back to original pixels, performs local Sobel edge snapping, and creates the crop and overlay.

Qwen is now a fallback for ambiguous visual cases rather than the first line of defense. Rectangle geometry, sharpness scoring, boundary composition, validation, refinement, diagnostic images, and pixel extraction are performed by Python/OpenCV/Pillow. These thresholds are initial heuristics, not trained confidence probabilities; they should be calibrated against labelled screenshots before being treated as general guarantees.

## Requirements

- 64-bit Python 3.12
- A current LM Studio version with the native REST chat API
- LM Studio installed
- A vision-language model downloaded in LM Studio
- The LM Studio `lms` CLI available on PATH if the script should start the server or load a model automatically

From this directory in PowerShell:

    .\setup.ps1

This creates a separate `.venv` and installs Requests, Pillow, NumPy, headless OpenCV, and pytest. It does not touch `experiments/sam3-python/.venv`.

## Model choice

Qwen3-VL-8B Instruct is the default recommendation. Qwen3-VL-8B Thinking is a separate reasoning-oriented checkpoint and may spend its output budget reasoning without returning the compact candidate selection needed by this experiment. Other vision-language Instruct models can also be selected explicitly, such as Qwen2.5-VL-3B Instruct.

The model must already be downloaded; the experiment does not fetch large model weights automatically.

## Run

With the default Instruct model downloaded in LM Studio:

    .\.venv\Scripts\python.exe -m inlay_qwen3vl C:\path\to\screenshot.jpg

The default API identifier is:

    qwen3vl-8b

For another local vision-language model:

    .\.venv\Scripts\python.exe -m inlay_qwen3vl screenshot.jpg --model qwen2.5-vl-3b-instruct --load-model qwen2.5-vl-3b-instruct

Useful options:

- `--model <identifier>` — API model identifier assigned to the loaded model
- `--load-model <exact-local-model-key>` — override automatic local-model discovery
- `--max-candidates 8` — maximum number of OpenCV proposals shown to Qwen
- `--output <directory>` — diagnostics output directory (default: <image-name>.qwen3vl)
- `--api-url http://localhost:1234/api/v1/chat` — LM Studio native chat endpoint
- `--max-image-dimension 2048` — maximum side length of images sent to Qwen

You can also configure INLAY_QWEN_MODEL, INLAY_LMSTUDIO_API_URL, INLAY_LMS_EXECUTABLE, and INLAY_LMS_LOAD_MODEL environment variables.

## Output

By default, the experiment creates a directory named `<image-name>.qwen3vl` beside the input image. It contains:

- `candidate.png` — selected and edge-refined photo candidate
- `overlay.png` — selected rectangle drawn over the original screenshot
- `proposals.png` — all numbered OpenCV candidates shown to Qwen
- `sharpness.png` — local detail-energy map; brighter areas are sharper and darker areas are blurrier
- `boundary_options.png` — four-panel view of the alternative left, right, top, and bottom edge positions
- `report.json` — candidate geometry scores, boundary contrast evidence, independently selected boundary IDs, selection method, refinement offsets, and output paths

The experiment deliberately does not create `crop.png` or mark a result accepted automatically. Inspect `overlay.png` and `candidate.png` before using the result. A good computer-vision score or model selection is not an accuracy guarantee.

## Tests

The tests use synthetic screenshots and mocked LM Studio responses. They do not require a running server or a downloaded model:

    .\.venv\Scripts\python.exe -m pytest -q

## Scope

This version uses classical OpenCV edge and contour analysis to generate rectangular proposals. It does not use a trained segmentation model or infer a pixel mask. Add more complex segmentation only if measured examples show that edge-based candidates are insufficient.
