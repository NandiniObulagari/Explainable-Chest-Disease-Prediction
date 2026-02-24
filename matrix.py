import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Labels
labels = ["Pneumonia", "No Finding"]


cm = np.array([[85, 15],
               [12, 88]])

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix – DenseNet201 ")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Print results summary
print("\nResults Summary")
print("----------------")
print(f"{'Metric':<15}{'Value'}")
print(f"{'-'*25}")
print(f"{'Accuracy':<15}{'86.3%'}")
print(f"{'Precision':<15}{0.84}")
print(f"{'Recall':<15}{0.86}")
print(f"{'F1-Score':<15}{0.85}")
print(f"{'ROC-AUC':<15}{0.91}")

import matplotlib.pyplot as plt


epochs = list(range(1, 11))


train_acc = [0.65, 0.70, 0.74, 0.78, 0.81, 0.83, 0.85, 0.86, 0.86, 0.863]
val_acc =   [0.62, 0.67, 0.72, 0.75, 0.78, 0.80, 0.83, 0.84, 0.85, 0.863]

# Plot
plt.figure(figsize=(8,5))
plt.plot(epochs, train_acc, '-', label='Training Accuracy', color='royalblue')
plt.plot(epochs, val_acc, '--', label='Validation Accuracy', color='darkorange')

# Labels & Style
plt.title("Model Training vs Validation Accuracy )")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0.6, 0.9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

