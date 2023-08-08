import os
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

from unet.unet_model import UNet  

def load_trained_model(weights_path):
    # Load the trained U-Net model
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(weights_path))
    model.eval()  
    return model

def preprocess_image(image_path):
    # Load and preprocess the input image for inference
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    preprocessed_image = transform(image).unsqueeze(0)  

    return image, preprocessed_image

def perform_inference(model, preprocessed_image):
    # Perform inference
    with torch.no_grad():
        model.eval()
        prediction = model(preprocessed_image)

    return prediction

def apply_threshold(prediction, threshold=0.5):
    # Apply thresholding to generate the binary mask
    binary_mask = (prediction > threshold).float()

    return binary_mask.squeeze(0)  

def overlay_masks_on_image(image, binary_mask, alpha=0.5):
    # Convert the binary mask to a 3-channel mask and convert to a Numpy array
    mask = binary_mask.expand(3, -1, -1)
    mask_p = mask.permute(1, 2, 0)
    mask_int = np.array(mask_p, dtype=np.uint8)

    # Convert the PIL image to a NumPy array and resize
    image_np = np.array(image)
    resized_image = cv2.resize(image_np, (256, 256))

    # Create the overlay by blending the original image and the mask
    overlay = cv2.addWeighted(resized_image, 1 - alpha, mask_int * 255 , alpha, 0)

    return overlay

if __name__ == "__main__":
    
    weights_path = os.path.join(os.getcwd(), "unet_model.pth")
    model = load_trained_model(weights_path)

    image_path = os.path.join(os.getcwd(), "dataset\\test\\images\\01_test.tif")
    original_image, preprocessed_image = preprocess_image(image_path)

    prediction = perform_inference(model, preprocessed_image)

    threshold = 0.5
    binary_mask = apply_threshold(prediction, threshold)

    alpha = 0.5
    overlay = overlay_masks_on_image(original_image, binary_mask, alpha)

    # Save the overlay as a new image
    save_path = os.path.join(os.getcwd(), "dataset\\results\\01_test_overlay.tif")
    cv2.imwrite(save_path, overlay)
