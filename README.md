# Oil_Spill_Segmentation

A web app that detects oil spills in images using deep learning. Upload a satellite or ocean surface image, and the app highlights exactly where the spill is by generating a segmentation mask and a red overlay on the original photo.

## What it does

Oil spills are usually spotted by manually inspecting satellite or aerial images, which is slow and error-prone. This project automates that step: a trained image segmentation model looks at an uploaded image and predicts, pixel by pixel, which areas are covered by an oil spill.

The result is returned as:
- A **binary mask** — a black-and-white image where white pixels mark the spill area
- A **red overlay** — the original image with the detected spill area tinted red, making it easy to see at a glance

## How it works

1. **Frontend (Flask + HTML template):** The user uploads an image through a simple web page.
2. **Backend (Flask route `/predict`):** The uploaded file is saved, then passed to the segmentation pipeline.
3. **Model:** A U-Net architecture (`segmentation_models_pytorch`) with a ResNet-34 encoder pretrained on ImageNet. The encoder extracts image features, and the decoder upsamples them back into a full-resolution mask predicting "spill" vs. "no spill" for every pixel.
4. **Inference:** The image is resized to 256×256, run through the model, passed through a sigmoid, and thresholded at 0.5 to produce a binary mask. The mask is then resized to 512×512 and blended with the original image to create the red overlay.
5. **Response:** The resulting image is sent back to the browser and displayed to the user.

## Tech stack

| Layer | Tools |
|---|---|
| Web framework | Flask, Gunicorn (for production) |
| Deep learning | PyTorch, `segmentation_models_pytorch` (U-Net + ResNet-34) |
| Image processing | Pillow, NumPy, torchvision transforms |
| Deployment | Configured for Render (`.render.yaml`, `Procfile`, `runtime.txt`) |

## Project structure

```
Oil_Spill_Segmentation/
├── app.py              # Flask app: routes for the homepage and prediction endpoint
├── inference.py        # Loads the model and runs segmentation on an uploaded image
├── model_arch.py        # Defines the U-Net (ResNet-34 encoder) model architecture
├── model.pth            # Trained model weights
├── templates/           # HTML page(s) for the web UI
├── static/               # Static assets + generated output images
├── uploads/              # Temporary storage for user-uploaded images
├── requirements.txt      # Python dependencies
├── Procfile              # Start command for deployment (gunicorn)
├── runtime.txt           # Python version for deployment
└── .render.yaml          # Render.com deployment config
```

## Getting started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/AfshaZareen/Oil_Spill_Segmentation.git
cd Oil_Spill_Segmentation
pip install -r requirements.txt
```

### Run locally

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser, upload an image, and view the predicted oil spill mask and overlay.

### Run in production

The app is set up to run with Gunicorn (as used on platforms like Render):

```bash
gunicorn app:app
```

## Model details

- **Architecture:** U-Net
- **Encoder:** ResNet-34, pretrained on ImageNet
- **Input:** RGB image, resized to 256×256
- **Output:** Single-channel mask (spill probability per pixel), thresholded at 0.5
- **Weights:** Loaded from `model.pth`

## Notes

- Predictions are only as good as the model's training data — results may vary on images very different from what the model was trained on (e.g. different lighting, resolution, or water conditions).
- Runs on CPU by default but will automatically use a GPU if one is available (`torch.cuda.is_available()`).
