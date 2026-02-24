import torch
import torch.nn as nn

class CombinedHeteroscedasticBCE(nn.Module):
    """
    Combines Binary Cross Entropy with a heteroscedastic uncertainty term.
    The model predicts both mu (mean logits) and log_var (log variance).
    """

    def __init__(self, alpha=0.6, beta=0.4):
        """
        Args:
            alpha (float): weight for BCE loss
            beta (float): weight for uncertainty loss
        """
        super(CombinedHeteroscedasticBCE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, mu, log_var, targets):
        """
        Args:
            mu (Tensor): mean logits predicted by the model [batch_size, num_classes]
            log_var (Tensor): predicted log variance [batch_size, num_classes]
            targets (Tensor): ground truth labels [batch_size, num_classes]
        """
        # 1️⃣ Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-5.0, max=5.0)

        # 2️⃣ Convert log_var → variance
        var = torch.exp(log_var)

        # 3️⃣ Apply reparameterization: add uncertainty to mu
        noise = torch.randn_like(mu)
        pred = mu + noise * torch.sqrt(var)

        # 4️⃣ Compute binary cross entropy between noisy logits and labels
        bce_loss = self.bce(pred, targets)

        # 5️⃣ Add regularization term — penalize large variance
        uncertainty_loss = torch.mean(var)

        # 6️⃣ Combine both losses
        total_loss = self.alpha * bce_loss + self.beta * uncertainty_loss
        return total_loss


# ----------------------------
# ✅ Test Function
# ----------------------------
if __name__ == "__main__":
    batch_size, num_classes = 4, 24
    mu = torch.randn(batch_size, num_classes)
    log_var = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()

    criterion = CombinedHeteroscedasticBCE(alpha=0.6, beta=0.4)
    loss = criterion(mu, log_var, targets)

    print("✅ Loss test successful!")
    print("Loss value:", loss.item())
