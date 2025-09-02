import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Sample confusion matrix data (based on your provided example)
cm = np.array([
    [200, 10, 5, 0, 0, 0, 0, 0, 0, 0],  # Healthy
    [15, 180, 8, 2, 0, 0, 0, 0, 0, 0],  # Bacterial_Spot
    [5, 7, 175, 8, 0, 0, 0, 0, 0, 0],   # Early_Blight
    [0, 2, 5, 190, 0, 0, 0, 0, 0, 0],   # Late_Blight
    [0, 0, 0, 0, 185, 5, 0, 0, 0, 0],   # Leaf_Mold
    [0, 0, 0, 0, 3, 180, 0, 0, 0, 2],   # Septoria_Leaf_Spot
    [0, 0, 0, 0, 0, 0, 195, 0, 0, 0],   # Spider_Mites
    [0, 0, 0, 0, 0, 0, 0, 190, 5, 0],   # Target_Spot
    [0, 0, 0, 0, 0, 0, 0, 3, 185, 0],   # Yellow_Leaf_Curl_Virus
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 183]    # Mosaic_Virus
])

# Define class names
class_names = ['Healthy', 'Bacterial_Spot', 'Early_Blight', 'Late_Blight', 
               'Leaf_Mold', 'Septoria_Leaf_Spot', 'Spider_Mites', 'Target_Spot', 
               'Yellow_Leaf_Curl_Virus', 'Mosaic_Virus']

# Calculate total accuracy
total_accuracy = np.trace(cm) / np.sum(cm)

# Create the plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix for MobileNetV3 Small\nTotal Accuracy: {total_accuracy:.2%}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save the image with a white background and black text
plt.savefig('confusion_matrix.png', dpi=300, facecolor='white', edgecolor='none')
plt.show()

print("Confusion matrix image saved as 'confusion_matrix.png'.")