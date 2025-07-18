# C-Section Wound Classification App

## Description
This project is a comprehensive wound classification system that combines a mobile application with advanced deep learning capabilities. The app allows medical professionals to capture or upload wound images and receive instant classifications of wound types (venous, diabetic, pressure, and surgical) using a trained deep learning model.

## GitHub Repository
[Link to GitHub Repository](https://github.com/Esther-Mbanzabigwi/wound_classification_Model.git)

## Environment Setup

### Python Environment (For Model Training)
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Mobile App Setup (React Native)
1. Prerequisites:
   - Node.js (v14 or higher)
   - Java Development Kit (JDK) 11
   - Android Studio with Android SDK
   - React Native CLI

2. Install dependencies:
```bash
cd WoundClassifierApp
npm install
```

3. Run the app:
```bash
# Start Metro bundler
npm start

# Run on Android
npm run android

# Build Android APK
npm run build:android
```

## Project Structure
```
F-c-section/
├── improved_wound_classification.ipynb  # Model training notebook
├── train_model.py                      # Model training script
├── wound_classifier_best.pth           # Trained model weights
├── requirements.txt                    # Python dependencies
└── WoundClassifierApp/                 # Mobile application
    ├── src/
    │   ├── components/                 # React components
    │   ├── screens/                    # App screens
    │   └── utils/                      # Utility functions
    └── android/                        # Android specific files
```

## App Interface screenshoots

1. Home Screen
   <img width="591" height="1280" alt="image" src="https://github.com/user-attachments/assets/3439791d-855e-48ff-a0cb-57998c0aa00d" />
   ![WhatsApp Image 2025-07-18 at 19 07 18](https://github.com/user-attachments/assets/6b5d192d-8759-4f88-9ff4-bd6c85db1c18)
   
    ![WhatsApp Image 2025-07-18 at 19 10 52](https://github.com/user-attachments/assets/eb68bdc7-3d5a-4d6b-977c-0b39e4880629)

2. Camera Screen
   ![WhatsApp Image 2025-07-18 at 19 11 09](https://github.com/user-attachments/assets/7fd72941-409b-45cb-b704-7769b4640464)

  
3. Results Screen 

   ![WhatsApp Image 2025-07-18 at 19 11 50](https://github.com/user-attachments/assets/df13ec9d-eea7-402a-8efd-e778a6dd360e)



## Deployment Plan

### Mobile App Deployment
1. **Testing Phase**
   - Internal testing with development team
   - Beta testing with selected medical professionals
   - Bug fixes and performance optimization

2. **Production Release**
   - Generate signed APK
   - Upload to Google Play Store
   - Monitor crash reports and user feedback

3. **Maintenance**
   - Regular updates for bug fixes
   - Model improvements
   - Feature enhancements based on user feedback

### Model Deployment
1. **Model Optimization**
   - Convert to TorchScript format
   - Quantization for mobile deployment
   - Performance testing on target devices

2. **Integration**
   - Embed model in mobile app
   - Implement version control for model updates
   - Setup monitoring for model performance

## Video Demo
[MotherMend Demo](https://youtu.be/JpcSdBfA39s)]

The video demonstration covers:
1. App Overview
2. Image Capture Process
3. Gallery Image Selection
4. Wound Classification Process
5. Results Interpretation
6. Additional Features

## Technical Details

### Model Architecture
- Based on state-of-the-art deep learning architecture
- Trained on the AZH dataset with 730 wound images
- Supports classification of four wound types:
  - Venous
  - Diabetic
  - Pressure
  - Surgical

### Mobile App Features
- Real-time wound image capture
- Gallery image selection
- Secure data handling
- Offline classification capability
- Result sharing functionality
- Historical record keeping

## Acknowledgments
Based on research work published in:
1. Patel, Y., et al. (2024). Integrated image and location analysis for wound classification: a deep learning approach. Scientific Reports, 14(1).
2. Anisuzzaman, D. M., et al. (2022). Multi-modal wound classification using wound image and location by deep neural network. Scientific Reports, 12(1). 
