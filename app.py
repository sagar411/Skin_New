from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# Initialize Flask app
app = Flask(__name__)

# Define the path to your saved model
MODEL_PATH = './model/best_skin_type_model.pth'

# Load the model architecture
def load_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 3)  # Assuming 3 classes
    
    # Determine the device to load the model on (CUDA if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the saved model state dict
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Remove fc layer weights from the state dict (as they won't match)
    checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith('fc.')}

    # Load the state dict into the model (ignoring the fc layer)
    model.load_state_dict(checkpoint, strict=False)

    model.eval()  # Set model to evaluation mode

    # Move the model to the appropriate device
    model.to(device)
    return model


# Load the model
model = load_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

index_label = {0: "dry", 1: "normal", 2: "oily"}

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Load and preprocess the image
        img = Image.open(file.stream).convert("RGB")
        img = transform(img).unsqueeze(0)  # Add batch dimension
        
        # Determine the device to run the model on
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move the image tensor to the appropriate device
        img = img.to(device)

        # Make prediction
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()

        # Return prediction
        return jsonify({'prediction': index_label[prediction]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app if this script is run directly
if __name__ == '__main__':
    app.run(debug=True)
