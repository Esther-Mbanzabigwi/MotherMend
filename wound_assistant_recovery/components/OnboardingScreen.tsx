import { Colors } from "@/constants/Colors";
import { useAuthContext } from "@/context/authcontext";
import { router } from "expo-router";
import {
    Dimensions,
    Image,
    StyleSheet,
    Text,
    TouchableOpacity,
    View,
} from "react-native";
import AppLayout from "./AppLayout";

const { width, height } = Dimensions.get("window");

interface OnboardingScreenProps {
  title: string;
  description: string;
  imageSource: any;
  currentPage: number;
  totalPages: number;
  onNext: () => void;
  onSkip?: () => void;
  showSkip?: boolean;
  isLastPage?: boolean;
}

export default function OnboardingScreen({
  title,
  description,
  imageSource,
  currentPage,
  totalPages,
  onNext,
  onSkip,
  showSkip = true,
  isLastPage = false,
}: OnboardingScreenProps) {
  const { completeOnboarding } = useAuthContext();

  const handleSkip = async () => {
    if (onSkip) {
      onSkip();
    } else {
      // Default skip behavior - mark onboarding complete and go to auth
      await completeOnboarding();
      router.replace("/(auth)");
    }
  };

  return (
    <AppLayout showIcon={false} scrollEnabled={false}>
      {/* Skip Button */}
      {showSkip && (
        <TouchableOpacity style={styles.skipButton} onPress={handleSkip}>
          <Text style={styles.skipText}>Skip</Text>
        </TouchableOpacity>
      )}

      {/* Progress Dots */}
      <View style={styles.progressContainer}>
        {Array.from({ length: totalPages }, (_, index) => (
          <View
            key={index}
            style={[
              styles.dot,
              index === currentPage - 1 ? styles.activeDot : styles.inactiveDot,
            ]}
          />
        ))}
      </View>

      {/* Image */}
      <View style={styles.imageContainer}>
        <Image source={imageSource} style={styles.image} resizeMode="contain" />
      </View>

      {/* Content */}
      <View style={styles.contentContainer}>
        <Text style={styles.title}>{title}</Text>
        <Text style={styles.description}>{description}</Text>
      </View>

      {/* Navigation Button */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.nextButton} onPress={onNext}>
          <Text style={styles.nextButtonText}>
            {isLastPage ? "Get Started" : "Next"}
          </Text>
        </TouchableOpacity>
      </View>
    </AppLayout>
  );
}

const styles = StyleSheet.create({
  skipButton: {
    position: "absolute",
    top: 20,
    right: 20,
    zIndex: 1,
  },
  skipText: {
    fontSize: 16,
    color: Colors.light.primary,
    fontWeight: "600",
  },
  progressContainer: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    marginTop: 40,
    marginBottom: 40,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginHorizontal: 4,
  },
  activeDot: {
    backgroundColor: Colors.light.primary,
  },
  inactiveDot: {
    backgroundColor: Colors.light.gray[300],
  },
  imageContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingHorizontal: 40,
  },
  image: {
    width: width * 0.7,
    height: height * 0.3,
  },
  contentContainer: {
    paddingHorizontal: 40,
    marginBottom: 40,
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    color: Colors.light.text,
    textAlign: "center",
    marginBottom: 16,
    lineHeight: 36,
  },
  description: {
    fontSize: 16,
    color: Colors.light.gray[600],
    textAlign: "center",
    lineHeight: 24,
  },
  buttonContainer: {
    paddingHorizontal: 40,
    marginBottom: 40,
  },
  nextButton: {
    backgroundColor: Colors.light.primary,
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 12,
    alignItems: "center",
  },
  nextButtonText: {
    color: "white",
    fontSize: 18,
    fontWeight: "600",
  },
});
