import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from torchvision import models

# Initialize Flask app
app = Flask(__name__)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet model from torchvision
model = models.efficientnet_b0(weights=None)  # Use weights=None instead of pretrained=False
num_features = model.classifier[1].in_features  # Get the number of input features
model.classifier[1] = nn.Linear(num_features, 2)  # Modify for binary classification

# Load trained model checkpoint
MODEL_PATH = "model_epoch_10.pth"  # Update with the correct path
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)  # Add weights_only=True
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')  # Serve index.html


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    image = Image.open(file).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        label = 'Fake' if pred.item() == 1 else 'Real'
    
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)