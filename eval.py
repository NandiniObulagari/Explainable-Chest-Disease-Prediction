import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from src.model import DenseNet201_Hetero, NUM_LABELS
from src.dataset import ChestXrayDataset, get_transforms
from src.loss import CombinedHeteroscedasticBCE


def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)

            mu, log_var = model(images)
            loss = criterion(mu, log_var, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(mu)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    return avg_loss, all_labels, all_preds


def compute_metrics(y_true, y_pred):
    """Compute multilabel classification metrics safely"""
    metrics = {}

    # ROC-AUC for multilabel
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        metrics["roc_auc"] = float("nan")

    # Convert continuous preds → binary (0 or 1)
    y_pred_binary = (y_pred >= 0.5).astype(int)

    # F1-score (macro across labels)
    try:
        metrics["f1"] = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
    except ValueError:
        metrics["f1"] = float("nan")

    # Accuracy (fraction of exact matches)
    metrics["accuracy"] = np.mean((y_true == y_pred_binary).astype(float))

    return metrics


def main():
    # =========================
    # CONFIG
    # =========================
    CSV_PATH = r"C:\Users\NANDINI\OneDrive\Desktop\dl\data\NIH\sample_labels.csv"
    IMG_DIR = r"C:\Users\NANDINI\OneDrive\Desktop\dl\data\NIH\sample\images"
    MODEL_PATH = r"C:\Users\NANDINI\OneDrive\Desktop\dl\models\densenet_hetero.pth"
    RESULTS_PATH = r"C:\Users\NANDINI\OneDrive\Desktop\dl\results\predictions.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8

    print("📦 Loading evaluation dataset...")
    dataset = ChestXrayDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=get_transforms(train=False))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("⚙ Loading model...")
    model = DenseNet201_Hetero(num_classes=NUM_LABELS, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    criterion = CombinedHeteroscedasticBCE(alpha=0.6, beta=0.4)

    # =========================
    # EVALUATION
    # =========================
    print("🚀 Starting evaluation...")
    val_loss, y_true, y_pred = evaluate(model, dataloader, criterion, DEVICE)
    metrics = compute_metrics(y_true, y_pred)

    print("\n📊 Evaluation Results:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    # =========================
    # SAVE PREDICTIONS
    # =========================
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    df = pd.DataFrame(y_pred, columns=[f"Label_{i}" for i in range(y_pred.shape[1])])
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\n💾 Predictions saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
