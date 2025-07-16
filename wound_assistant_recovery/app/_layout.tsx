import { Colors } from "@/constants/Colors";
import AuthContextProvider from "@/context/authcontext";
import { Stack } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { SafeAreaView } from "react-native";

export default function RootLayout() {
  return (
    <AuthContextProvider>
      <SafeAreaView
        style={{ flex: 1, backgroundColor: Colors.light.background }}
      >
        <StatusBar style="dark" />
        <Stack
          screenOptions={{
            headerShown: false,
          }}
        >
          <Stack.Screen
            name="(auth)"
            options={{
              headerShown: false,
            }}
          />
          <Stack.Screen
            name="(tabs)"
            options={{
              headerShown: false,
            }}
          />
        </Stack>
      </SafeAreaView>
    </AuthContextProvider>
  );
}
