from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# Initialize Flask app
app = Flask(__name__)

# Define the path to your saved model
MODEL_PATH = './model/best_skin_type_model.pth'

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model architecture and weights
def load_model():
    # Load ResNet50 with pretrained weights from ImageNet
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # Modify the fully connected layer to match the number of classes (3 classes)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 3)  # 3 output classes (dry, normal, oily)

    # Determine the device to load the model on (CUDA if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the saved model state dict
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Remove the fc layer weights from the checkpoint since they don't match the new model's fc
    checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith('fc.')}

    # Load the rest of the state dict into the model
    model.load_state_dict(checkpoint, strict=False)  # strict=False allows missing keys (fc layer)

    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move model to the appropriate device (CPU or CUDA)
    
    return model


# Load the model
best_model = load_model()

# Skin type labels
index_label = {0: "dry", 1: "normal", 2: "oily"}

# Prediction function for single image
def predict(x):
    # Load and preprocess the image
    img = Image.open(x).convert("RGB")
    img = transform(img).unsqueeze(0)  # Apply transformation and add batch dimension

    # Use the appropriate device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the image tensor to the appropriate device
    img = img.to(device)

    # Move the model to the appropriate device
    best_model.to(device)

    # Set the model to evaluation mode and disable gradient calculation for inference
    best_model.eval()
    with torch.no_grad():
        # Pass the image through the model
        out = best_model(img)
        
        # Get the predicted class (argmax)
        return out.argmax(1).item()

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Make prediction
        prediction_index = predict(file)
        prediction_label = index_label[prediction_index]

        # Return prediction
        return jsonify({'prediction': prediction_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app if this script is run directly
if __name__ == '__main__':
    app.run(debug=True)
