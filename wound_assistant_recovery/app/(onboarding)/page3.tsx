import OnboardingScreen from "@/components/OnboardingScreen";
import { useAuthContext } from "@/context/authcontext";
import { router } from "expo-router";

export default function OnboardingPage3() {
  const { completeOnboarding } = useAuthContext();

  const handleNext = () => {
    router.push("/(onboarding)/page4");
  };

  const handleSkip = async () => {
    await completeOnboarding();
    router.replace("/(auth)");
  };

  return (
    <OnboardingScreen
      title="Organization & Health Tracking"
      description="Track your wound recovery, monitor pain levels, set medication reminders, and stay informed with healing tips."
      imageSource={require("@/assets/images/mother2.png")}
      currentPage={3}
      totalPages={4}
      onNext={handleNext}
      onSkip={handleSkip}
    />
  );
}
