import os
from PIL import Image
import numpy as np


def foreground(data_folder):
    # Create output directory to save the foreground images
    output_dir = os.path.join(data_folder, "foreground_images")
    os.makedirs(output_dir, exist_ok=True)

    # Assuming images and masks are stored in separate folders within data_folder
    images_folder = os.path.join(data_folder, "images")
    masks_folder = os.path.join(data_folder, "mask")

    def get_mask_filename(image_filename):
        mask_filename = image_filename.split(".")[0] + "_mask.gif"  # Assumes image_001.jpg -> mask_001.jpg
        return mask_filename

    image_filenames = os.listdir(images_folder)
    mask_filenames = os.listdir(masks_folder)


    for image_filename in image_filenames:
        image_path = os.path.join(images_folder, image_filename)
        mask_filename = get_mask_filename(image_filename)
        mask_path = os.path.join(masks_folder, mask_filename)

        # Load image and mask as NumPy arrays
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Create binary mask
        binary_mask = (mask > 0).astype(np.uint8)

        # Apply binary mask to the original image
        foreground = image * np.expand_dims(binary_mask, axis=-1)

        # Convert the foreground array back to an image and save it
        foreground_image = Image.fromarray(foreground)
        foreground_image.save(os.path.join(output_dir, image_filename))