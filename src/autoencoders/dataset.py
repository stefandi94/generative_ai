import glob

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

from torchvision.transforms import v2

from consts import ANIMAL_FACES_LABELS

class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = glob.glob(root_dir + "/*/**.jpg", recursive=True)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        label = self.get_label(self.image_files[idx])

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def get_label(image_path):
        for key in ANIMAL_FACES_LABELS:
            if key in image_path:
                return ANIMAL_FACES_LABELS[key]
        raise Exception(f'No valid label found! Available labels: {list(ANIMAL_FACES_LABELS.values())}')



class AnimalDataLoader:
    def __init__(self, root_dir, batch_size=32, shuffle=True, num_workers=0):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            v2.Resize((64, 64)),        # Resize images to 64x64
            v2.ToTensor(),              # Convert images to PyTorch tensors
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.dataset = AnimalDataset(root_dir=self.root_dir, transform=self.transform)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                      shuffle=self.shuffle, num_workers=self.num_workers)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return iter(self.dataloader)


# Example usage:
if __name__ == "__main__":
    from consts import ANIMAL_FACES_TRAIN_DIR

    # Initialize ImageDataLoader
    data_loader = AnimalDataLoader(root_dir=ANIMAL_FACES_TRAIN_DIR, batch_size=32, shuffle=True, num_workers=4)

    # Iterate over batches
    for batch in data_loader:
        # Your training loop here
        print("Batch of images shape:", batch.shape)  # Shape: (batch_size, channels, height, width)
