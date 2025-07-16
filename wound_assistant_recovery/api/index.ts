import axios from "axios";
import { ImagePickerAsset } from "expo-image-picker";

const modelUrl = "https://wound-model-api.onrender.com";

// Types
export interface WoundPrediction {
  predicted_class: string;
  confidence: number;
  urgency_level: "HIGH" | "MEDIUM" | "LOW";
  requires_hospital: boolean;
  recommendations: string[];
  class_probabilities: Record<string, number>;
}

// API Service
const api = {
  classifyWound: async (
    imageUri: ImagePickerAsset
  ): Promise<WoundPrediction> => {
    const formData = new FormData();
    console.log("imageUri ================== =");
    formData.append("file", imageUri as unknown as Blob);
    console.log("formData ================== =");

    const response = await axios.post(`${modelUrl}/predict`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    console.log("response ================== =");

    return response.data;
  },
};

export default api;
