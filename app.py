from flask import Flask, request, jsonify
from inference import load_trained_model, preprocess_image, perform_inference, apply_threshold, overlay_masks_on_image
import glob
import os
from PIL import Image
import cv2

app = Flask(__name__)

# Load the trained U-Net model 
#weights_path = "C:\\Users\\ImaniM\\coding\\pytorch_unet\\unet_model.pth"
weights_path = os.path.join(os.getcwd(), "unet_model.pth")
model = load_trained_model(weights_path)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the path to the image directory 
    images_directory = request.form.get("images_directory")

    # Get a list of image files in the directory
    image_files = glob.glob(os.path.join(images_directory, "*.tif"))

    predictions = []

    for image_file in image_files:
        # Load and preprocess the input image for inference
        _, preprocessed_image = preprocess_image(image_file)

        # Perform inference
        prediction = perform_inference(model, preprocessed_image)

        # Apply thresholding to generate the binary mask
        threshold = 0.5
        binary_mask = apply_threshold(prediction, threshold)

        # Overlay the segmentation mask on the original image
        alpha = 0.5
        overlay = overlay_masks_on_image(Image.open(image_file), binary_mask, alpha)

        # Save the overlay as a new image 
        overlay_name = os.path.basename(image_file).replace(".tif", "_overlay.tif")
        save_directory = os.path.join(os.getcwd(), "dataset\\results")
        os.makedirs(save_directory, exist_ok=True)
        overlay_path = os.path.join(save_directory, overlay_name)
        cv2.imwrite(overlay_path, overlay)

        predictions.append({
            'image_path': image_file,
            'prediction': binary_mask.tolist(),
            'overlay_path': overlay_path
        })
        
    return jsonify(predictions)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
