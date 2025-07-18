# MotherMend – C-Section Wound Monitoring System

> **An AI-powered mobile application for early detection of post-cesarean wound infections in Rwanda.**

MotherMend leverages **React Native**, **FastAPI/Strapi**, and a **MobileNetV2-based CNN** to classify C-section wound images into six categories (Healthy Tissue, Eschar, Granulating Tissue, Necrotic Tissue, Slough, Undefined). It provides **real-time feedback**, **offline functionality**, and a **secure data pipeline**, enabling mothers and CHWs to monitor wounds and prevent surgical site infections (SSIs).

---

## **Table of Contents**
1. [Key Features](#key-features)  
2. [System Architecture](#system-architecture)  
3. [Tech Stack](#tech-stack)  
4. [Installation Guide](#installation-guide)  
5. [How to Run the App](#how-to-run-the-app)  
6. [Testing and Demo](#testing-and-demo)  
7. [Performance Metrics](#performance-metrics)  
8. [Results & Analysis](#results--analysis)  
9. [Deployment](#deployment)  
10. [Screenshots](#screenshots)  
11. [Recommendations & Future Work](#recommendations--future-work)  
12. [License](#license)

---

## **Key Features**
- **AI-Powered Wound Classification**  
  - MobileNetV2 CNN model with **87.5% accuracy** and **AUC-ROC of 0.91**.
- **Offline and Online Modes**  
  - TensorFlow Lite inference on-device.
- **Secure Data Handling**  
  - **AES-256 encryption**, **JWT-based authentication**, and HTTPS.
- **Multilingual UI**  
  - English, Kinyarwanda, and French support.
- **Low-Latency Predictions**  
  - **<5 seconds response time**, even on low-end smartphones.
- **User-Centric Design**  
  - Voice-assisted guidance and visual cues for low-literacy users.

---

## **System Architecture**
- **Frontend:** React Native (Expo)  
- **Backend:** FastAPI + Strapi CMS  
- **Database:** PostgreSQL (or SQLite for local testing)  
- **AI Model:** MobileNetV2 CNN deployed via TFLite  
- **Hosting:** AWS/Google Cloud for model APIs

**Flow:**  
`User → Mobile App → Backend API → AI Model → Feedback & Records`

---

## **Tech Stack**
- **Mobile App:** React Native, Expo, TypeScript  
- **Backend & APIs:** FastAPI, Strapi, Python  
- **AI Model:** TensorFlow, Keras, MobileNetV2  
- **Database:** PostgreSQL / SQLite  
- **Cloud Services:** Firebase, AWS S3 (optional)

---

## **Installation Guide**

### **Prerequisites**
- Node.js (v18 or later)
- Python 3.11+
- Git
- Expo CLI (`npm install -g expo-cli`)
- Virtual environment for Python (optional but recommended)

---

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/Esther-Mbanzabigwi/MotherMend.git
cd MotherMend
```

### **Step 2: Setup Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
- By default, the API runs on `http://localhost:8502`

### **Step 3: Setup Mobile App**
```bash
cd app
npm install
npx expo start
```
- Scan the QR code with Expo Go (iOS/Android) to test the app.

---

## **How to Run the App**
1. Start the backend server (`uvicorn main:app --reload`).  
2. Run the React Native app with Expo (`npx expo start`).  
3. Use the camera/gallery to upload wound images.  
4. View classification results and confidence scores.

---

## **Testing and Demo**
- **Testing Strategies:**  
  - Unit testing for AI inference.  
  - Functional testing for core features.  
  - Cross-device performance testing (low-end vs. high-end smartphones).  
  - Testing with **different data values** and simulated lighting conditions.  

- **Demo Video:**  
  [**Watch Demo (5 min)**](https://youtu.be/YsZlm6oCXKk)

---

## **Performance Metrics**
| **Metric**               | **Value**        |
|--------------------------|-----------------:|
| Model Accuracy           | **87.52%**       |
| Precision                | **86.10%**       |
| Recall                   | **82.45%**       |
| F1-Score                 | **84.24%**       |
| AUC-ROC                  | **0.91**         |
| Average Prediction Time  | **4.5 sec**      |
| Uptime (Testing)         | **99.2%**        |
| User Ease-of-Use Rating  | **4/5**          |

---

## **Results & Analysis**
The MotherMend system successfully achieved its main and specific objectives:  
- **AI Model Performance:** The MobileNetV2 CNN achieved **87.52% accuracy** with strong precision and recall, meeting the target of >85%. Grad-CAM visualizations validated model interpretability, highlighting key wound regions.  
- **App Performance:** Prediction latency averaged **4.5 seconds**, within the <5s requirement, ensuring smooth user experience even on low-end hardware.  
- **Testing with Different Devices:** The system performed consistently across both mid-tier and low-end Android devices, with negligible differences in inference time (4.5 – 6s).  
- **Usability Feedback:** Pilot users (mothers and CHWs) rated the app **4/5 for ease of use**, with **85% reporting clear understanding of results**.  
- **Gap Analysis:** While most objectives were met, improvements such as **voice-guided instructions** and **further dataset expansion** were identified for future versions.

---

## **Deployment**
- **Deployed Model:** [**Download Here**](http://127.0.0.1:8502)  
- **API Documentation:** [**Swagger UI**](http://127.0.0.1:8502/docs)  
- **Repository:** [**GitHub Repo**](https://github.com/Esther-Mbanzabigwi/MotherMend)

---

## **Screenshots**
(Add these screenshots in the repo's `/screenshots` folder)  
1. **Home Screen**
   ![WhatsApp Image 2025-07-18 at 19 10 52](https://github.com/user-attachments/assets/7615e802-f58e-42e2-a6e6-bf2057e3148d)    ![WhatsApp Image 2025-07-18 at 19 07 18](https://github.com/user-attachments/assets/38a05664-ab73-4f96-ad34-84560c9b5c17)


2. **Upload Screen**
   ![WhatsApp Image 2025-07-18 at 19 11 09](https://github.com/user-attachments/assets/9c415864-8603-42c8-b74f-2ba72b9b1951)

3. **Prediction Results**

    <img width="327" height="475" alt="image" src="https://github.com/user-attachments/assets/c4a389ea-fc35-4c12-aad4-e0aade6d669d" />

4. **History/Records Page**

   ![WhatsApp Image 2025-07-18 at 19 11 50](https://github.com/user-attachments/assets/c33a0366-0054-49aa-b777-5df6bacde3ab)


---

## **Recommendations & Future Work**
- **Expand Dataset:** Add more expert-labeled wound images from Rwandan hospitals.  
- **Voice & Accessibility Features:** Add voice-guided instructions for low-literacy users.  
- **Integration with Health Systems:** Connect to Rwanda’s RapidSMS and health platforms.  
- **Optimize Offline Mode:** Reduce TFLite model size for faster inference on low-end devices.  
- **Scalability:** Conduct larger pilot testing across multiple rural districts.

---

## **License**
This project is developed as part of the **BSc. Software Engineering Capstone Project** and is protected under the MIT License.
