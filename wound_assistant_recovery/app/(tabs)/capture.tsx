import * as ImagePicker from "expo-image-picker";
import { useEffect, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  StyleSheet,
  Text,
  View,
} from "react-native";
import api from "../../api";
import AppLayout from "../../components/AppLayout";
import Button from "../../components/ui/Button";
import { Colors } from "../../constants/Colors";

export default function CaptureScreen() {
  const [image, setImage] = useState<ImagePicker.ImagePickerAsset | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    (async () => {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== "granted") {
        Alert.alert(
          "Permission needed",
          "Camera permission is required to take photos."
        );
      }
    })();
  }, []);

  const analyzeWound = async () => {
    try {
      setLoading(true);
      const prediction = await api.classifyWound(image!);
      console.log("prediction ================== =");
      console.log(prediction);
    } catch (error) {
      console.log(error);
      Alert.alert(
        "Error",
        "Failed to analyze wound. Please check your internet connection and try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const takePhoto = async () => {
    try {
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled) {
        setImage(result.assets[0]);
        analyzeWound();
      }
    } catch (error) {
      console.log(error);
      Alert.alert("Error", "Failed to take photo. Please try again.");
    }
  };

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled) {
        setImage(result.assets[0]);
        analyzeWound();
      }
    } catch (error) {
      console.log(error);
      Alert.alert("Error", "Failed to pick image. Please try again.");
    }
  };

  return (
    <AppLayout scrollEnabled={false}>
      <View style={styles.captureArea}>
        {image ? (
          <View style={styles.imagePreview}>
            <Image source={{ uri: image.uri }} style={styles.previewImage} />
          </View>
        ) : (
          <View style={styles.cameraPlaceholder}>
            <Text style={styles.cameraIcon}>📸</Text>
            <Text style={styles.captureText}>Track My Recovery</Text>
          </View>
        )}

        {loading ? (
          <View style={styles.loading}>
            <ActivityIndicator size="large" color={Colors.light.primary} />
            <Text style={styles.loadingText}>Processing your photo...</Text>
          </View>
        ) : (
          <View style={styles.actions}>
            <Button
              title="Take Photo"
              variant="primary"
              onPress={takePhoto}
              style={styles.button}
            />
            <Text style={styles.orText}>or</Text>
            <Button
              title="Choose from Gallery"
              variant="secondary"
              onPress={pickImage}
              style={styles.button}
            />
          </View>
        )}
      </View>
    </AppLayout>
  );
}

const styles = StyleSheet.create({
  captureArea: {
    flex: 1,
    justifyContent: "center",
    gap: 32,
  },
  cameraPlaceholder: {
    backgroundColor: Colors.light.card,
    borderRadius: 20,
    padding: 40,
    alignItems: "center",
    justifyContent: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  cameraIcon: {
    fontSize: 48,
    marginBottom: 16,
  },
  captureText: {
    fontSize: 18,
    color: Colors.light.primary,
    fontWeight: "600",
    textAlign: "center",
  },
  imagePreview: {
    backgroundColor: Colors.light.card,
    borderRadius: 20,
    overflow: "hidden",
    aspectRatio: 4 / 3,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  previewImage: {
    width: "100%",
    height: "100%",
  },
  actions: {
    gap: 16,
    alignItems: "center",
  },
  button: {
    width: "100%",
  },
  orText: {
    color: Colors.light.gray[500],
    fontSize: 16,
  },
  loading: {
    alignItems: "center",
    gap: 16,
  },
  loadingText: {
    fontSize: 16,
    color: Colors.light.gray[500],
  },
});
