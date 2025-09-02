import torch
from torchvision import models

# Number of classes (same as your trained model)
num_classes = 10

# Load MobileNetV3 model architecture
model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)

# Load trained weights
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
model.eval()

# Create dummy input to simulate an image
dummy_input = torch.randn(1, 3, 224, 224)

# Convert model to TorchScript using tracing
traced_script_module = torch.jit.trace(model, dummy_input)

# Save the traced model
traced_script_module.save("plant_disease_model.pt")

print("✅ TorchScript model saved as plant_disease_model.pt")
