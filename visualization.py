import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients
from tqdm import tqdm

from src.dataset import ChestXrayDataset, get_transforms
from src.model import DenseNet201_Hetero, NUM_LABELS


# ----------------------------
# WRAPPER MODEL FOR CAPTUM
# ----------------------------
class ModelWrapper(torch.nn.Module):
    """Captum-compatible wrapper for your model that returns only mu."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        mu, _ = self.model(x)
        return mu


# ----------------------------
# VISUALIZATION FUNCTION
# ----------------------------
def visualize_integrated_gradients(model, dataloader, label_names, device, save_dir="outputs/visualizations"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    print("🧠 Generating explainability maps using Integrated Gradients...")

    wrapped_model = ModelWrapper(model)
    ig = IntegratedGradients(wrapped_model)

    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Explaining batches")):
        images, labels = images.to(device), labels.to(device)
        mu, _ = model(images)
        preds = torch.sigmoid(mu)

        for i in range(min(4, images.size(0))):  # visualize first few samples per batch
            input_img = images[i].unsqueeze(0)
            true_labels = labels[i].cpu().numpy()
            pred_labels = preds[i].detach().cpu().numpy()

            # Find most probable label index
            target_idx = int(np.argmax(pred_labels))

            # Compute attributions for that label
            attributions, delta = ig.attribute(input_img, target=target_idx, return_convergence_delta=True)
            attr = attributions.squeeze().detach().cpu().numpy()
            img = input_img.squeeze().detach().cpu().permute(1, 2, 0).numpy()

            # Normalize for display
            attr_norm = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Build readable label lists
            true_label_names = [label_names[j] for j, val in enumerate(true_labels) if val > 0.5]
            top_preds = np.argsort(pred_labels)[::-1][:3]
            pred_label_names = [f"{label_names[j]} ({pred_labels[j]:.2f})" for j in top_preds]

            # Plot original and IG overlay
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].imshow(img_norm)
            ax[0].set_title("Original Image")
            ax[0].axis("off")

            ax[1].imshow(img_norm, alpha=0.6)
            ax[1].imshow(np.mean(attr_norm, axis=0), cmap="jet", alpha=0.5)
            ax[1].set_title("Integrated Gradients")
            ax[1].axis("off")

            # Add text info below plots
            plt.suptitle(
                f"True: {', '.join(true_label_names) if true_label_names else 'No Finding'}\n"
                f"Pred: {', '.join(pred_label_names)}",
                fontsize=10, y=0.02
            )

            # Save result
            save_path = os.path.join(save_dir, f"explain_b{batch_idx}_i{i}.png")
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.savefig(save_path)
            plt.close(fig)

            print(f"✅ Saved visualization → {save_path}")

        # Limit to first 2 batches for speed
        if batch_idx >= 1:
            break

    print(f"\n🎨 All visualizations saved to: {save_dir}")


# ----------------------------
# MAIN EXECUTION
# ----------------------------
def main():
    # =========================
    # CONFIG
    # =========================
    CSV_PATH = r"C:\Users\NANDINI\OneDrive\Desktop\dl\data\NIH\sample_labels.csv"
    IMG_DIR = r"C:\Users\NANDINI\OneDrive\Desktop\dl\data\NIH\sample\images"
    MODEL_PATH = r"C:\Users\NANDINI\OneDrive\Desktop\dl\models\densenet_hetero.pth"
    SAVE_DIR = r"C:\Users\NANDINI\OneDrive\Desktop\dl\outputs\visualizations"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4

    # =========================
    # LOAD DATA
    # =========================
    print("📦 Loading dataset for visualization...")
    dataset = ChestXrayDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=get_transforms(train=False))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Extract label names (excluding meta columns)
    label_names = [c for c in dataset.data.columns if c not in ['Image Index', 'Finding Labels', 'img_path']]

    # =========================
    # LOAD MODEL
    # =========================
    print("⚙ Loading trained model...")
    model = DenseNet201_Hetero(num_classes=len(label_names), pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # =========================
    # RUN VISUALIZATION
    # =========================
    visualize_integrated_gradients(model, dataloader, label_names, DEVICE, save_dir=SAVE_DIR)


if __name__ == "__main__":
    main()
