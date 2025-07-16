import { Redirect } from "expo-router";
import { ActivityIndicator, StyleSheet, View } from "react-native";
import { Colors } from "../constants/Colors";
import { useAuthContext } from "../context/authcontext";

export default function IndexScreen() {
  const { user, token, isInitialized, hasSeenOnboarding } = useAuthContext();

  // Show loading while auth context is initializing
  if (!isInitialized) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color={Colors.light.primary} />
      </View>
    );
  }

  // Use Redirect component for navigation
  if (user && token) {
    return <Redirect href="/(tabs)" />;
  } else if (!hasSeenOnboarding) {
    return <Redirect href="/(onboarding)/page1" />;
  } else {
    return <Redirect href="/(auth)" />;
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: Colors.light.background,
  },
});
