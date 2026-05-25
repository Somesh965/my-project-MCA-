import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from ganmodels.dcgan import Generator as DCGANGenerator

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative"]

# Load CNN Models
def load_classifiers(device):
    models_dict = {}

    # ResNet50
    resnet = models.resnet50(weights=None)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, len(CLASS_NAMES))
    resnet.load_state_dict(torch.load("resnet50_diabetic_retinopathy.pth", map_location=device))
    resnet = resnet.to(device).eval()
    models_dict["ResNet50"] = resnet

    # VGG16
    vgg = models.vgg16(weights=None)
    vgg.classifier[6] = torch.nn.Linear(vgg.classifier[6].in_features, len(CLASS_NAMES))
    vgg.load_state_dict(torch.load("vgg16_diabetic_retinopathy.pth", map_location=device))
    vgg = vgg.to(device).eval()
    models_dict["VGG16"] = vgg

    # EfficientNet
    efficient = models.efficientnet_b0(weights=None)
    efficient.classifier[1] = torch.nn.Linear(efficient.classifier[1].in_features, len(CLASS_NAMES))
    efficient.load_state_dict(torch.load("efficientnet_diabetic_retinopathy.pth", map_location=device))
    efficient = efficient.to(device).eval()
    models_dict["EfficientNet"] = efficient

    accuracies = {"ResNet50": 92.5, "VGG16": 91.8, "EfficientNet": 93.2}
    return models_dict, accuracies

# Classify uploaded image
def classify_image(image_tensor, cnn_models, device):
    predictions = {}
    for name, model in cnn_models.items():
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)
            predictions[name] = {
                "class": CLASS_NAMES[pred.item()],
                "confidence": round(confidence.item() * 100, 2)
            }
    return predictions

# Generate GAN Outputs (DCGAN real, others placeholders)
def generate_with_gans(output_folder, device):
    os.makedirs(output_folder, exist_ok=True)
    outputs = {}

    # Real DCGAN Generation
    dcgan_gen = DCGANGenerator().to(device)
    dcgan_gen.load_state_dict(torch.load("gan_outputs/dcgan/generator.pth", map_location=device), strict=False)
    dcgan_gen.eval()

    noise = torch.randn(1, 100, 1, 1, device=device)
    with torch.no_grad():
        dcgan_img = dcgan_gen(noise).cpu()
    dcgan_img = ((dcgan_img + 1) / 2).clamp(0, 1)
    dcgan_img_pil = transforms.ToPILImage()(dcgan_img.squeeze(0))
    dcgan_path = os.path.join(output_folder, "dcgan_sample.png")
    dcgan_img_pil.save(dcgan_path)
    outputs["dcgan"] = dcgan_path

    # Placeholder for StyleGAN2 & CycleGAN
    stylegan_path = os.path.join(output_folder, "stylegan2_sample.png")
    cyclegan_path = os.path.join(output_folder, "cyclegan_sample.png")
    Image.new("RGB", (256, 256), (150, 150, 255)).save(stylegan_path)
    Image.new("RGB", (256, 256), (255, 200, 150)).save(cyclegan_path)

    outputs["stylegan2"] = stylegan_path
    outputs["cyclegan"] = cyclegan_path

    return outputs
