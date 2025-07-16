import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Label mapping
label_map = {
    0: "Eschar",
    1: "Granulating Tissue",
    2: "Healthy Tissue",
    3: "Necrotic Tissue", 
    4: "Slough",
    5: "Undefined"
}

# Model definition
class ImprovedWoundClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ImprovedWoundClassifier, self).__init__()
        
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedWoundClassifier()
model.to(device)
checkpoint = torch.load('wound_classifier_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test a single image
def analyze_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get model predictions and probabilities
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    # Get sorted probabilities and classes
    probs, indices = torch.sort(probabilities[0], descending=True)
    
    print(f"\nAnalysis for image: {image_path}")
    print("\nPredictions (sorted by confidence):")
    for prob, idx in zip(probs, indices):
        print(f"{label_map[idx.item()]}: {prob.item()*100:.2f}%")

# Test directory
test_dir = "wound-classification-using-images-and-locations-main/wound-classification-using-images-and-locations-main/dataset/Test"

# Test some images from different classes
for class_name in ['D', 'N', 'P', 'S', 'V']:
    class_dir = os.path.join(test_dir, class_name)
    if os.path.exists(class_dir):
        # Get first image from each class
        images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        if images:
            image_path = os.path.join(class_dir, images[0])
            analyze_image(image_path)

print("\nThis analysis helps identify if the model is consistently biased towards the Slough class.") 