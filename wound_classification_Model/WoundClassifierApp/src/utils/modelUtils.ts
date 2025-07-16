import { MobileModel, media, torch } from 'react-native-pytorch-core';

const MODEL_URL = 'wound_classifier_best.pth';
const LABELS = ['BG', 'D', 'N', 'P', 'S', 'V'];
const LABEL_DESCRIPTIONS = {
  BG: 'Background',
  D: 'Dehiscence',
  N: 'Necrosis',
  P: 'Pressure Injury',
  S: 'Surgical Site Infection',
  V: 'Various',
};

const RECOMMENDATIONS = {
  BG: [],
  D: [
    'Keep the wound clean and covered',
    'Avoid physical strain on the affected area',
    'Schedule follow-up with healthcare provider',
    'Monitor for any increased separation',
  ],
  N: [
    'Urgent medical attention required',
    'Keep the area clean and protected',
    'Monitor for any changes in color or odor',
    'Follow prescribed treatment plan',
  ],
  P: [
    'Relieve pressure on the affected area',
    'Maintain proper positioning',
    'Keep the wound clean and dressed',
    'Monitor for signs of infection',
  ],
  S: [
    'Keep the wound clean and dry',
    'Monitor for increased redness or swelling',
    'Take prescribed antibiotics as directed',
    'Contact healthcare provider if condition worsens',
  ],
  V: [
    'Follow specific care instructions',
    'Keep the wound clean',
    'Monitor for any changes',
    'Regular follow-up with healthcare provider',
  ],
};

export interface ClassificationResult {
  type: string;
  confidence: number;
  recommendations: string[];
}

export const loadModel = async (): Promise<MobileModel> => {
  try {
    const model = await MobileModel.loadModel(MODEL_URL);
    return model;
  } catch (error) {
    throw new Error('Failed to load model: ' + error);
  }
};

export const preprocessImage = async (imageUri: string) => {
  try {
    // Load the image
    const image = await media.fromURL(imageUri);
    
    // Resize to model input size (224x224)
    const resizedImage = await image.resize(224, 224);
    
    // Convert to tensor and normalize
    const tensor = await torch.fromBlob(
      resizedImage.getBytes(),
      [1, 3, 224, 224],
      'float32'
    );
    
    // Normalize the tensor (similar to what we did in training)
    const normalized = tensor.div(255.0);
    
    return normalized;
  } catch (error) {
    throw new Error('Failed to preprocess image: ' + error);
  }
};

export const classifyImage = async (
  model: MobileModel,
  imageUri: string
): Promise<ClassificationResult> => {
  try {
    // Preprocess the image
    const inputTensor = await preprocessImage(imageUri);
    
    // Run inference
    const output = await model.forward(inputTensor);
    
    // Get prediction
    const scores = await output.softmax();
    const prediction = await scores.argmax();
    const confidence = await scores.max();
    
    // Get the predicted label
    const labelIndex = await prediction.item();
    const label = LABELS[labelIndex];
    const type = LABEL_DESCRIPTIONS[label];
    
    return {
      type,
      confidence: confidence,
      recommendations: RECOMMENDATIONS[label],
    };
  } catch (error) {
    throw new Error('Failed to classify image: ' + error);
  }
};

export const cleanupModel = async (model: MobileModel) => {
  try {
    await model.destroy();
  } catch (error) {
    console.error('Error cleaning up model:', error);
  }
}; 