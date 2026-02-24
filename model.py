import torch
import torch.nn as nn
import torchvision.models as models

# ----------------------------
# CONSTANTS
# ----------------------------
NUM_LABELS = 24 # ✅ Must match number of labels your dataset printed in dataset.py

# ----------------------------
# MODEL DEFINITION
# ----------------------------
class DenseNet201_Hetero(nn.Module):
    def __init__(self, num_classes=NUM_LABELS, pretrained=True):
        super(DenseNet201_Hetero, self).__init__()

        # Load pretrained DenseNet201
        densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None)

        # Remove the classifier layer (we’ll replace it)
        self.features = densenet.features
        self.num_features = densenet.classifier.in_features

        # Two parallel fully connected layers for heteroscedastic outputs
        self.fc_mu = nn.Linear(self.num_features, num_classes)
        self.fc_log_var = nn.Linear(self.num_features, num_classes)

        # Activation for uncertainty estimation stability
        self.softplus = nn.Softplus()

    def forward(self, x):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        # Predict mean and log variance for each label
        mu = self.fc_mu(out)
        log_var = self.fc_log_var(out)

        # Optional: constrain log_var for numerical stability
        log_var = torch.clamp(log_var, min=-5, max=5)

        return mu, log_var


# ----------------------------
# Test Model Shape
# ----------------------------
if __name__ == "__main__":
    model = DenseNet201_Hetero(num_classes=NUM_LABELS)
    x = torch.randn(2, 3, 224, 224)
    mu, log_var = model(x)
    print("✅ Model test successful!")
    print("Mu shape:", mu.shape)
    print("LogVar shape:", log_var.shape)
