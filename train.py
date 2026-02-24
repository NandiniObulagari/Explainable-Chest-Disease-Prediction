import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.model import DenseNet201_Hetero, NUM_LABELS  # ✅ correct class name
from src.loss import CombinedHeteroscedasticBCE
from src.dataset import ChestXrayDataset, get_transforms


def main():
    # ===========================
    # CONFIGURATION
    # ===========================
    CSV_PATH = r"C:\Users\NANDINI\OneDrive\Desktop\dl\data\NIH\sample_labels.csv"
    IMG_DIR = r"C:\Users\NANDINI\OneDrive\Desktop\dl\data\NIH\sample\images"
    SAVE_PATH = r"C:\Users\NANDINI\OneDrive\Desktop\dl\models\densenet_hetero.pth"

    BATCH_SIZE = 8
    EPOCHS = 5
    LR = 1e-4
    VAL_SPLIT = 0.2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # ===========================
    # DATASET + DATALOADERS
    # ===========================
    print("📦 Loading dataset...")
    dataset = ChestXrayDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=get_transforms(train=True))

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # ⚠ On Windows: num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ===========================
    # MODEL + LOSS + OPTIMIZER
    # ===========================
    print("⚙ Initializing model...")
    model = DenseNet201_Hetero(num_classes=NUM_LABELS, pretrained=True).to(DEVICE)
    criterion = CombinedHeteroscedasticBCE(alpha=0.6, beta=0.4)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    # ===========================
    # TRAINING LOOP
    # ===========================
    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            mu, log_var = model(images)
            loss = criterion(mu, log_var, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # ===========================
        # VALIDATION LOOP
        # ===========================
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                mu, log_var = model(images)
                loss = criterion(mu, log_var, labels)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"✅ Epoch {epoch+1} Summary → Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ===========================
        # SAVE BEST MODEL
        # ===========================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"💾 Saved best model → {SAVE_PATH}")

    print("🎉 Training complete!")


if __name__ == "__main__":
    main()
