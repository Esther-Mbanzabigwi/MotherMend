import { Platform } from 'react-native';

// Server URL - use the same for both development and production
const API_URL = 'http://192.168.1.68:8080';

export interface ClassificationResponse {
  predicted_class: string;
  confidence: number;
  urgency_level: 'HIGH' | 'MEDIUM' | 'LOW';
  requires_hospital: boolean;
  recommendations: string[];
}

export const classifyWound = async (imageUri: string): Promise<ClassificationResponse> => {
  try {
    // Create form data
    const formData = new FormData();
    formData.append('file', {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'wound_image.jpg',
    } as any); // Use type assertion to fix FormData typing issue

    // Make API call
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error classifying wound:', error);
    throw error;
  }
};

export const checkServerHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_URL}/health`);
    return response.ok;
  } catch (error) {
    console.error('Server health check failed:', error);
    return false;
  }
}; 