import os
from sklearn.model_selection import train_test_split


def train_val_split(data_folder, train_size=0.8, random_state=1):

    image_folder = os.path.join(data_folder, "images")
    mask_folder = os.path.join(data_folder, "1st_manual")

    image_filenames = os.listdir(image_folder)
    mask_filenames = os.listdir(mask_folder)

    # Remove file extensions to match image and mask filenames
    image_ids = [filename.split("_")[0] for filename in image_filenames]
    mask_ids = [filename.split("_")[0] for filename in mask_filenames]

    # Find common image-mask pairs
    common_ids = list(set(image_ids) & set(mask_ids))

    # Split the common ids into training and validation sets
    train_ids, val_ids = train_test_split(common_ids, train_size=train_size, random_state=random_state)

    return train_ids, val_ids
