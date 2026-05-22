# Unified OCR API (Handwriting + General OCR)

An end-to-end line-level OCR service combining a custom **CRNN** (with optimized C++ ONNX Runtime option) for handwriting recognition and **Tesseract** for general printed text fallback. Includes full-page preprocessing (deskew + custom line segmentation) and single-line inference.

---

## Key Features

- **Dynamic Pipeline**: Choose between optimized ONNX (`crnn_onnx`), PyTorch (`crnn`), and `tesseract` on the fly.
- **Page Preprocessing**: Automated line segmentation, deskewing, and adaptive contrast normalization.
- **Production Ready**: Run locally or via a microservice-ready Docker container.
- **Super Light Docker Footprint**: Leverages conditional dependencies to strip container size to a bare minimum.

---

## ⚡ Quick Start with Docker

1. Place your compiled ONNX model at `model_weights/handwriting_recognizer_best.onnx`. *(Optionally, place the PyTorch checkpoint at `model_weights/handwriting_recognizer_best.pth`)*
2. Spin up the API service:
   ```bash
   docker compose up --build
   ```
3. Test the API:
   ```bash
   curl http://localhost:8000/health
   ```
4. Explore interactive API docs at `http://localhost:8000/docs`.

---

## 🔄 Architecture & Internal Flow of Execution

For a detailed phase-by-phase breakdown of system startup, preprocessing algorithms, dynamic model routing, and character decoding, check out the dedicated technical blueprint:

👉 **[Internal Execution Flow & Architecture Documentation](execution_flow.md)**

---

## 🐳 Docker Image Size Optimization (Massive Space Savings)

We have heavily optimized our Docker workflow to create a fast, secure, and production-grade environment. By focusing on minimal dependencies, we achieved major space reductions:

### 🛡️ Optimization Techniques Used:
- **Headless OpenCV**: Swapped generic OpenCV for `opencv-python-headless`, removing heavy graphical dependencies like X11, OpenGL, and display libraries.
- **Decoupled Video Codecs**: Completely purged the heavy `libheif-dev` package (which pulls in H.265, H.264, and AV1 codecs) saving several hundred megabytes.
- **No-Install-Recommends**: Used `--no-install-recommends` during Tesseract system setup, preventing bloated language packs and auxiliary packages from being installed.
- **Zero-PyTorch Mode**: Allowed booting without PyTorch (`DEVICE_TYPE=none`) to run inference purely through C++ optimized ONNX runtime.

## 🛠️ Local Installation

### 1. Prerequisites
- **Python 3.10+**
- **Tesseract OCR**
  - *Windows*: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki/Downloads) and add to system `PATH`.
  - *macOS*: `brew install tesseract`
  - *Linux*: `sudo apt-get install tesseract-ocr`

### 2. Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m api.fastapi_app
```

---

## 🌐 API Reference

| Method | Endpoint | Description | Query Parameters |
| :--- | :--- | :--- | :--- |
| **`GET`** | `/health` | Live status & active hardware device | None |
| **`GET`** | `/models` | Available model parameters and metadata | None |
| **`POST`**| `/ocr` | OCR multiple images concurrently | `model_type` (`crnn_onnx` \| `crnn` \| `tesseract`), `preprocessing_mode` (`full` \| `single_line`) |
| **`POST`**| `/ocr/single`| OCR a single image (Swagger UI friendly) | `model_type`, `preprocessing_mode` |

### Example cURL Request:
```bash
curl -X POST "http://localhost:8000/ocr/single?model_type=crnn_onnx&preprocessing_mode=full" \
  -F "file=@IMG_20260412_070458.jpg"
```

---

## 📂 Project Structure

```
ocr-merged/
├── api/                 # FastAPI configuration, routes, and JSON schemas
├── models/              # ONNX & PyTorch CRNN adapters, Tesseract bindings, model registry
├── preprocessing/       # Full-page deskewing, line segmentation, and normalization
├── scripts/             # Auxiliary developer utility scripts (e.g. ONNX exporting)
├── model_weights/       # Directory to mount/copy .pth and .onnx weight files (gitignored)
├── notebooks/           # Model training, experiments, and reinforcement learning
├── results/             # Example API response payload files
├── Dockerfile           # Optimized multi-mode Docker configuration
├── docker-compose.yml   # Volume and environment setup for Docker runs
├── requirements.txt     # Local development and notebook dependencies
└── requirements-docker.txt # Production runtime dependencies (no PyTorch)
```

---

## 🔧 Troubleshooting

- **CRNN "weights not found"**: Place your `handwriting_recognizer_best.onnx` or `handwriting_recognizer_best.pth` file into the `model_weights/` directory.
- **Tesseract errors**: Double-check that Tesseract is installed and on your system's `PATH` variable when running locally.
- **Performance tuning**: For fast local CPU execution, default to `model_type=crnn_onnx` to leverage fast C++ inference engine pipelines instead of loading heavy PyTorch graphs.
