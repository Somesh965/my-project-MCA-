import os
import uuid
from flask import Flask, render_template, request
from utils import load_classifiers, classify_image, generate_with_gans
from PIL import Image
import torch
from torchvision import transforms

# Paths
UPLOAD_FOLDER = 'static/uploads'
GAN_OUTPUT_FOLDER = 'static/outputs/gan'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GAN_OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CNN Models
cnn_models, model_accuracies = load_classifiers(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = {}
    uploaded_image_path = None
    generated_images = {}

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html')

        if file:
            filename = str(uuid.uuid4()) + "_" + file.filename
            uploaded_image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(uploaded_image_path)

            # Preprocess image
            img = Image.open(uploaded_image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            # CNN Predictions
            predictions = classify_image(img_tensor, cnn_models, device)

            # GAN Outputs (Real DCGAN + placeholders)
            generated_images = generate_with_gans(GAN_OUTPUT_FOLDER, device)

    return render_template('index.html',
                           uploaded_image=uploaded_image_path,
                           predictions=predictions,
                           accuracies=model_accuracies,
                           generated_images=generated_images)

if __name__ == '__main__':
    app.run(debug=True)
