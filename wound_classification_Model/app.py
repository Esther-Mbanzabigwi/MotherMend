from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Wound Classification API",
    description="""
    This API provides wound classification services with the following features:
    - Wound type classification
    - Urgency assessment
    - Hospital visit recommendations
    - Care instructions
    
    Available endpoints:
    - /docs: Interactive API documentation (Swagger UI)
    - /redoc: Alternative API documentation
    - /: Web interface for wound classification
    - /predict/: Main classification endpoint
    - /health: Server health check
    """,
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the templates directory
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

# Configure templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Label mapping
label_map = {
    0: "Eschar",
    1: "Granulating Tissue",
    2: "Healthy Tissue",
    3: "Necrotic Tissue", 
    4: "Slough",
    5: "Undefined"
}

# Urgency and recommendations mapping
urgency_map = {
    "Eschar": {
        "urgency": "HIGH",
        "hospital_visit": True,
        "recommendations": [
            "Seek immediate medical attention",
            "Do not attempt to remove the eschar",
            "Keep the wound clean and covered",
            "Document any changes in wound appearance"
        ]
    },
    "Granulating Tissue": {
        "urgency": "LOW",
        "hospital_visit": False,
        "recommendations": [
            "Continue prescribed wound care",
            "Keep the area clean and moist",
            "Monitor for signs of infection",
            "Follow up with healthcare provider as scheduled"
        ]
    },
    "Healthy Tissue": {
        "urgency": "LOW",
        "hospital_visit": False,
        "recommendations": [
            "Continue normal skin care routine",
            "Protect the area from injury",
            "Monitor for any changes",
            "Regular check-ups as recommended"
        ]
    },
    "Necrotic Tissue": {
        "urgency": "HIGH",
        "hospital_visit": True,
        "recommendations": [
            "Seek immediate medical attention",
            "Do not attempt self-treatment",
            "Keep the wound covered",
            "Document the affected area"
        ]
    },
    "Slough": {
        "urgency": "MEDIUM",
        "hospital_visit": True,
        "recommendations": [
            "Schedule urgent medical appointment",
            "Keep the wound clean",
            "Monitor for spreading or increased pain",
            "Follow current wound care instructions"
        ]
    },
    "Undefined": {
        "urgency": "MEDIUM",
        "hospital_visit": True,
        "recommendations": [
            "Consult healthcare provider for proper assessment",
            "Keep the area clean and protected",
            "Document any changes or symptoms",
            "Avoid self-treatment without proper diagnosis"
        ]
    }
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model definition
class ImprovedWoundClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ImprovedWoundClassifier, self).__init__()
        
        # Use pretrained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=False)
        
        # Modify final layers
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
model = ImprovedWoundClassifier()
model.to(device)
checkpoint = torch.load('wound_classifier_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
async def read_root(request: Request):
    """
    Serves the web interface for wound classification.
    Access this through a web browser.
    """
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        print(f"Error serving template: {str(e)}")
        return HTMLResponse(content=f"Error: {str(e)}", status_code=500)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Save original image size for debugging
        original_size = image.size
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get predicted class label
        predicted_label = label_map[predicted_class]
        
        # Get urgency assessment and recommendations
        urgency_info = urgency_map[predicted_label]
        
        # Get all class probabilities
        class_probabilities = {
            label_map[i]: float(prob) 
            for i, prob in enumerate(probabilities[0].cpu().numpy())
        }
        
        # Sort probabilities for better visualization
        sorted_probs = dict(sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "predicted_class": predicted_label,
            "confidence": confidence,
            "urgency_level": urgency_info["urgency"],
            "requires_hospital": urgency_info["hospital_visit"],
            "recommendations": urgency_info["recommendations"],
            "class_probabilities": sorted_probs,
            "debug_info": {
                "original_image_size": original_size,
                "model_device": str(device),
                "input_tensor_shape": list(image_tensor.shape),
                "input_tensor_range": {
                    "min": float(image_tensor.min()),
                    "max": float(image_tensor.max()),
                    "mean": float(image_tensor.mean()),
                    "std": float(image_tensor.std())
                }
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/health")
async def health_check():
    """
    Check if the server is running and the model is loaded.
    Returns:
        dict: Status information about the server
    """
    return {
        "status": "ok",
        "message": "Service is running",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.get("/api-info")
async def api_info():
    """
    Get information about available endpoints and how to use them.
    Returns:
        dict: API usage information
    """
    return {
        "endpoints": {
            "/": "Web interface - open in browser",
            "/docs": "Interactive API documentation (Swagger UI)",
            "/redoc": "Alternative API documentation",
            "/predict": "POST endpoint for wound classification",
            "/health": "Server health check",
            "/api-info": "This information"
        },
        "mobile_usage": {
            "base_url": f"http://<your-ip-address>:8080",
            "example": "http://192.168.1.68:8080",
            "note": "Replace <your-ip-address> with your computer's IP address"
        },
        "supported_image_formats": ["jpg", "jpeg", "png"],
        "response_format": {
            "predicted_class": "string - wound type",
            "confidence": "float - prediction confidence (0-1)",
            "urgency_level": "string - HIGH/MEDIUM/LOW",
            "requires_hospital": "boolean - whether immediate medical attention is needed",
            "recommendations": "array of strings - care instructions"
        }
    }

@app.get("/docs-redirect")
async def docs_redirect():
    """
    Redirect to the API documentation
    """
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    from pathlib import Path
    import uvicorn

    # Define and check template path
    TEMPLATES_DIR = Path("templates")
    print(f"Templates directory: {TEMPLATES_DIR.resolve()}")
    print(f"Templates directory exists: {TEMPLATES_DIR.exists()}")
    print(f"Index.html exists: {(TEMPLATES_DIR / 'index.html').exists()}")

    print("\n=== Wound Classification API ===\n")
    print("Server running at:")
    print("✓ Web Interface & API: http://192.168.1.68:8502\n")
    print("Use this URL for both web browser and mobile app:")
    print("1. Web Browser: Open http://192.168.1.68:8502")
    print("2. Mobile App: Use http://192.168.1.68:8502/predict for classification\n")
    print("Available endpoints:")
    print("- /            : Web interface")
    print("- /predict     : Classification endpoint (POST)")
    print("- /docs        : API documentation")
    print("- /health      : Server status\n")
    print("Make sure your phone is connected to the same WiFi network!")

    uvicorn.run(app, host="127.0.0.1", port=8502)
