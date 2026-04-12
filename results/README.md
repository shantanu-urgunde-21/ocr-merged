# Portfolio samples

This folder holds **illustrative outputs** for recruiters and reviewers: example JSON shapes the API returns for each model. They are not live API dumps; replace or extend them with your own screenshots or responses if you want a richer demo.

## What the project does

- **Handwriting (CRNN):** A PyTorch CRNN trained on line-level IAM-style data, exposed through the same service as classical OCR.
- **Printed / general text (Tesseract):** Line segmentation (deskew + horizontal projection) plus Tesseract per line, selectable per request.
- **Unified API:** FastAPI with `model_type=crnn|tesseract` and `preprocessing_mode=full|single_line`.
- **Deployment:** Docker image installs runtime deps only; **CRNN weights are mounted** at `./model_weights` (see main README).

## Files

| File | Purpose |
|------|--------|
| [`sample_response_crnn.json`](sample_response_crnn.json) | Example `POST /ocr` response for CRNN, single-line mode. |
| [`sample_response_tesseract.json`](sample_response_tesseract.json) | Example response for Tesseract after full-page preprocessing. |

## Optional additions

Add your own **input/output image pairs** or a screenshot of `/docs` here to strengthen the portfolio; keep assets small so the repo stays clone-friendly.
