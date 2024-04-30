import torch
from tqdm import tqdm

from src.autoencoders import AutoEncoder, AnimalDataLoader

from consts import ANIMAL_FACES_TRAIN_DIR, ANIMAL_FACES_VAL_DIR


if __name__ == '__main__':
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    epochs = 200

    # Initialize ImageDataLoader
    train_data_loader = AnimalDataLoader(root_dir=ANIMAL_FACES_TRAIN_DIR, batch_size=8, shuffle=True, num_workers=8)
    valid_data_loader = AnimalDataLoader(root_dir=ANIMAL_FACES_VAL_DIR, batch_size=8, num_workers=8)
    autoencoder = AutoEncoder().to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters())
    loss_f = torch.nn.CrossEntropyLoss().to(device)
    # loss_f = torch.nn.MSELoss().to(device)

    for epoch in tqdm(range(epochs), desc="Epoch loop"):
        epoch_loss = 0
        # for images, _ in tqdm(train_data_loader, desc="Batch loop", total=len(train_data_loader)):
        for images, _ in train_data_loader:
            images = images.to(device)
            reconstructed_images = autoencoder(images)
            current_loss = loss_f(images, reconstructed_images)

            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
            epoch_loss += current_loss.item()

        epoch_loss /= len(train_data_loader)
        print(f"Loss: {epoch_loss:.2f}")
    print()