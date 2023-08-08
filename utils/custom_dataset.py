import os
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_folder, image_ids, mask_ids, transform=None):
        self.data_folder = data_folder
        self.image_ids = image_ids
        self.mask_ids = mask_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        mask_id = self.mask_ids[idx]

        image_path = os.path.join(self.data_folder, "images", f"{image_id}_training.tif")
        mask_path = os.path.join(self.data_folder, "1st_manual", f"{mask_id}_manual1.gif")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
