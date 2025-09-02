import torch
from torchvision import models
import torch.onnx
import tensorflow as tf
import subprocess

# Load model
model = models.mobilenet_v3_small()
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 10)  # 10 classes
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "plant_disease_model.onnx", opset_version=12)
print("ONNX model saved.")

# Convert ONNX to TensorFlow using subprocess
try:
    result = subprocess.run(
        ["python", "-m", "tf2onnx.convert", "--input", "plant_disease_model.onnx", 
         "--output", "plant_disease_model.pb", "--opset", "12"],
        check=True,
        capture_output=True,
        text=True
    )
    print("ONNX converted to TensorFlow model.")
    print(result.stdout)  # Print output for debugging
except subprocess.CalledProcessError as e:
    print(f"Error during conversion: {e}")
    print(f"Error output: {e.output}")
    exit(1)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("plant_disease_model.pb")
tflite_model = converter.convert()
with open("plant_disease_model.tflite", "wb") as f:
    f.write(tflite_model)
    print("TFLite model saved as plant_disease_model.tflite")