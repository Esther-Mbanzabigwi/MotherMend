import { useAuthContext } from "@/context/authcontext";
import { Link, router } from "expo-router";
import { useState } from "react";
import {
    Alert,
    Image,
    ScrollView,
    StyleSheet,
    Text,
    TouchableOpacity,
    View
} from "react-native";
import Button from "../../components/ui/Button";
import Input from "../../components/ui/Input";
import { Colors } from "../../constants/Colors";
import { SharedStyles } from "../../constants/SharedStyles";

export default function RegisterScreen() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const { register } = useAuthContext();
  
  // Form validation function
  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    // Full name validation
    if (!fullName.trim()) {
      newErrors.fullName = "Full name is required";
    } else if (fullName.trim().length < 2) {
      newErrors.fullName = "Full name must be at least 2 characters";
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email.trim()) {
      newErrors.email = "Email is required";
    } else if (!emailRegex.test(email)) {
      newErrors.email = "Please enter a valid email address";
    }

    // Phone number validation
    if (!phoneNumber.trim()) {
      newErrors.phoneNumber = "Phone number is required";
    } else if (phoneNumber.trim().length < 10) {
      newErrors.phoneNumber = "Please enter a valid phone number";
    }

    // Password validation
    if (!password) {
      newErrors.password = "Password is required";
    } else if (password.length < 6) {
      newErrors.password = "Password must be at least 6 characters";
    }

    // Confirm password validation
    if (!confirmPassword) {
      newErrors.confirmPassword = "Please confirm your password";
    } else if (password !== confirmPassword) {
      newErrors.confirmPassword = "Passwords do not match";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleRegister = async () => {
    if (!validateForm()) {
      return;
    }

    setIsLoading(true);
    try {
      // Then use the auth context to handle the user session
      await register({
        username: fullName.trim(),
        email: email.trim(),
        phone: phoneNumber.trim(),
        password: password,
      });

      // Show success message and navigate
      Alert.alert(
        "Success!",
        "Account created successfully. Welcome to WoundTrack!",
        [
          {
            text: "Continue",
            onPress: () => router.replace("/(tabs)"),
          },
        ]
      );
    } catch (error: any) {
      let errorMessage = "Registration failed. Please try again.";

      if (error.response?.data?.error?.message) {
        errorMessage = error.response.data.error.message;
      } else if (error.message) {
        errorMessage = error.message;
      }

      Alert.alert("Registration Error", errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // Clear error when user starts typing
  const clearError = (field: string) => {
    if (errors[field]) {
      setErrors((prev) => ({ ...prev, [field]: "" }));
    }
  };

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scrollContent}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <TouchableOpacity>
          <Link href="/" style={styles.backLink}>
            <Text style={styles.backText}>← Back to Home</Text>
          </Link>
        </TouchableOpacity>

        <View
          style={[
            SharedStyles.card,
            styles.authCard,
          ]}
        >
          <View style={styles.logo}>
            <View style={styles.iconContainer}>
              <Image
                source={require("../../assets/images/baby_hand.png")}
                style={styles.iconImage}
                resizeMode="cover"
              />
            </View>
            <Text style={[SharedStyles.title, styles.welcomeText]}>Join WoundTrack</Text>
            <Text style={SharedStyles.subtitle}>
              Create your account to start monitoring your recovery
            </Text>
          </View>
          
          <View style={styles.form}>
            <View style={SharedStyles.formGroup}>
              <Text style={SharedStyles.label}>Full Name</Text>
              <Input
                placeholder="Your full name"
                value={fullName}
                onChangeText={(text) => {
                  setFullName(text);
                  clearError("fullName");
                }}
                icon="user"
                error={errors.fullName}
              />
              {errors.fullName && (
                <Text style={SharedStyles.errorText}>{errors.fullName}</Text>
              )}
            </View>

            <View style={SharedStyles.formGroup}>
              <Text style={SharedStyles.label}>Email</Text>
              <Input
                placeholder="your.email@example.com"
                value={email}
                onChangeText={(text) => {
                  setEmail(text);
                  clearError("email");
                }}
                keyboardType="email-address"
                autoCapitalize="none"
                autoCorrect={false}
                icon="mail"
                error={errors.email}
              />
              {errors.email && (
                <Text style={SharedStyles.errorText}>{errors.email}</Text>
              )}
            </View>

            <View style={SharedStyles.formGroup}>
              <Text style={SharedStyles.label}>Phone Number</Text>
              <Input
                placeholder="+ (250) 000-000-000"
                value={phoneNumber}
                onChangeText={(text) => {
                  setPhoneNumber(text);
                  clearError("phoneNumber");
                }}
                keyboardType="phone-pad"
                icon="phone"
                error={errors.phoneNumber}
              />
              {errors.phoneNumber && (
                <Text style={SharedStyles.errorText}>{errors.phoneNumber}</Text>
              )}
            </View>

            <View style={SharedStyles.formGroup}>
              <Text style={SharedStyles.label}>Password</Text>
              <Input
                placeholder="••••••••"
                value={password}
                onChangeText={(text) => {
                  setPassword(text);
                  clearError("password");
                }}
                secureTextEntry
                autoCapitalize="none"
                autoCorrect={false}
                icon="lock"
                error={errors.password}
              />
              {errors.password && (
                <Text style={SharedStyles.errorText}>{errors.password}</Text>
              )}
            </View>

            <View style={SharedStyles.formGroup}>
              <Text style={SharedStyles.label}>Confirm Password</Text>
              <Input
                placeholder="••••••••"
                value={confirmPassword}
                onChangeText={(text) => {
                  setConfirmPassword(text);
                  clearError("confirmPassword");
                }}
                secureTextEntry
                autoCapitalize="none"
                autoCorrect={false}
                icon="lock"
                error={errors.confirmPassword}
              />
              {errors.confirmPassword && (
                <Text style={SharedStyles.errorText}>{errors.confirmPassword}</Text>
              )}
            </View>

            <Button
              variant="primary"
              onPress={handleRegister}
              style={styles.button}
              title={isLoading ? "Creating Account..." : "Create Account"}
              disabled={isLoading}
            />

            <View style={styles.footer}>
              <Text style={styles.footerText}>Already have an account? </Text>
              <Link href="/(auth)/login">
                <Text style={styles.footerLink}>Sign in here</Text>
              </Link>
            </View>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.light.background,
  },
  scrollContent: {
    padding: 20,
    paddingTop: 40,
    minHeight: "100%",
  },
  backLink: {
    marginBottom: 20,
    padding: 8,
  },
  backText: {
    color: Colors.light.primary,
    fontSize: 18,
    fontWeight: "500",
  },
  authCard: {
    backgroundColor: Colors.light.card,
    padding: 28,
    borderColor: Colors.light.border,
    borderWidth: 1,
    borderRadius: 24,
    shadowColor: Colors.light.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 12,
    elevation: 5,
  },
  logo: {
    alignItems: "center",
    marginBottom: 32,
  },
  iconContainer: {
    width: 90,
    height: 90,
    borderRadius: 45,
    backgroundColor: Colors.light.blue[500],
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 20,
    shadowColor: Colors.light.primary,
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.15,
    shadowRadius: 6,
    elevation: 3,
    overflow: "hidden",
    borderColor: Colors.light.blue[400],
    borderWidth: 2,
  },
  iconImage: {
    width: "80%",
    height: "80%",
    opacity: 1,
  },
  welcomeText: {
    color: Colors.light.primary,
    marginBottom: 8,
  },
  form: {
    gap: 24,
  },
  button: {
    marginTop: 8,
    borderRadius: 16,
    height: 56,
  },
  footer: {
    flexDirection: "row",
    justifyContent: "center",
    marginTop: 20,
    paddingVertical: 8,
  },
  footerText: {
    color: Colors.light.gray[500],
    fontSize: 15,
  },
  footerLink: {
    color: Colors.light.primary,
    fontWeight: "600",
    fontSize: 15,
  },
});
