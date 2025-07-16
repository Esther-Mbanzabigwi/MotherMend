import { Colors } from "@/constants/Colors";
import { Stack } from "expo-router";

export default function OnboardingLayout() {
  return (
    <Stack
      screenOptions={{
        headerShown: false,
        contentStyle: {
          backgroundColor: Colors.light.background,
        },
        animation: "slide_from_right",
      }}
    >
      <Stack.Screen
        name="page1"
        options={{
          headerShown: false,
        }}
      />
      <Stack.Screen
        name="page2"
        options={{
          headerShown: false,
        }}
      />
      <Stack.Screen
        name="page3"
        options={{
          headerShown: false,
        }}
      />
      <Stack.Screen
        name="page4"
        options={{
          headerShown: false,
        }}
      />
    </Stack>
  );
}
