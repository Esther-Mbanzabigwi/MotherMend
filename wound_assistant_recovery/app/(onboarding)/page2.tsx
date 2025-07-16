import OnboardingScreen from "@/components/OnboardingScreen";
import { useAuthContext } from "@/context/authcontext";
import { router } from "expo-router";

export default function OnboardingPage2() {
  const { completeOnboarding } = useAuthContext();

  const handleNext = () => {
    router.push("/(onboarding)/page3");
  };

  const handleSkip = async () => {
    await completeOnboarding();
    router.replace("/(auth)");
  };

  return (
    <OnboardingScreen
      title="Community & Support"
      description="Connect with fellow moms for support and advice."
      imageSource={require("@/assets/images/mother1.png")}
      currentPage={2}
      totalPages={4}
      onNext={handleNext}
      onSkip={handleSkip}
    />
  );
}
