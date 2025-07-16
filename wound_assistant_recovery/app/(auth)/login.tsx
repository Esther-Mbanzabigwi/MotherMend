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
    View,
} from "react-native";
import Button from "../../components/ui/Button";
import Input from "../../components/ui/Input";
import { Colors } from "../../constants/Colors";
import { SharedStyles } from "../../constants/SharedStyles";

export default function LoginScreen() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const { login } = useAuthContext();

  // Form validation function
  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email.trim()) {
      newErrors.email = "Email is required";
    } else if (!emailRegex.test(email)) {
      newErrors.email = "Please enter a valid email address";
    }

    // Password validation
    if (!password) {
      newErrors.password = "Password is required";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleLogin = async () => {
    if (!validateForm()) {
      return;
    }

    setIsLoading(true);
    try {
      const loginData = {
        identifier: email.trim(),
        password: password,
      };

      await login(loginData);

      router.replace("/(tabs)");
    } catch (error: any) {
      console.log(JSON.stringify(error));
      let errorMessage =
        "Login failed. Please check your credentials and try again.";

      if (error.response?.data?.error?.message) {
        errorMessage = error.response.data.error.message;
      } else if (error.message) {
        errorMessage = error.message;
      }

      Alert.alert("Login Error", errorMessage);
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
        style={SharedStyles.container}
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
            <Text style={[SharedStyles.title, styles.welcomeText]}>Welcome Back</Text>
            <Text style={SharedStyles.subtitle}>
              Continue your recovery journey
            </Text>
          </View>

            <View style={styles.form}>
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
                  icon="mail"
                  error={errors.email}
                />
                {errors.email && (
                  <Text style={SharedStyles.errorText}>{errors.email}</Text>
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
                  icon="lock"
                  error={errors.password}
                />
                {errors.password && (
                  <Text style={SharedStyles.errorText}>{errors.password}</Text>
                )}
                <TouchableOpacity style={styles.forgotPassword}>
                  <Text style={styles.forgotPasswordText}>
                    Forgot Password?
                  </Text>
                </TouchableOpacity>
              </View>

              <Button
                variant="primary"
                onPress={handleLogin}
                style={styles.button}
                title={isLoading ? "Signing In..." : "Sign In"}
                disabled={isLoading}
              />

              <View style={styles.footer}>
                <Text style={styles.footerText}>
                  Don&apos;t have an account?{" "}
                </Text>
                <Link href="/(auth)/register">
                  <Text style={styles.footerLink}>Sign up here</Text>
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
  forgotPassword: {
    alignSelf: "flex-end",
    marginTop: 8,
  },
  forgotPasswordText: {
    color: Colors.light.primary,
    fontSize: 14,
    fontWeight: "500",
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
