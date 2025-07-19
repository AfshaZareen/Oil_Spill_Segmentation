import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import os
from model_arch import get_model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = get_model()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# Fixed output size
OUTPUT_SIZE = (512, 512)

# Transform input image
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

def run_segmentation(image_path, output_mask_path="static/output_mask.png", overlay_path="static/output_overlay.png"):
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        output = model(input_tensor)[0][0]  # Shape: [H, W]

    # Sigmoid + threshold
    output = torch.sigmoid(output).cpu().numpy()
    binary_mask = (output > 0.5).astype(np.uint8) * 255  # [0,255]

    # Save mask image with fixed size
    mask_img = Image.fromarray(binary_mask).convert("L")
    mask_img = mask_img.resize(OUTPUT_SIZE)

    os.makedirs("static", exist_ok=True)
    mask_img.save(output_mask_path)

    # Resize original image to match fixed overlay size
    resized_image = image.resize(OUTPUT_SIZE)
    orig_np = np.array(resized_image)
    mask_resized = np.array(mask_img)

    # Create red overlay
    red_overlay = np.zeros_like(orig_np)
    red_overlay[:, :, 0] = mask_resized  # Red channel only

    overlay_np = np.clip(orig_np * 0.7 + red_overlay * 0.3, 0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(overlay_np)
    overlay_img.save(overlay_path)

    return output_mask_path, overlay_path
