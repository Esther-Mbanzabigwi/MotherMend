import OnboardingScreen from "@/components/OnboardingScreen";
import { useAuthContext } from "@/context/authcontext";
import { router } from "expo-router";

export default function OnboardingPage1() {
  const { completeOnboarding } = useAuthContext();

  const handleNext = () => {
    router.push("/(onboarding)/page2");
  };

  const handleSkip = async () => {
    await completeOnboarding();
    router.replace("/(auth)");
  };

  return (
    <OnboardingScreen
      title="Expert Support Introduction"
      description="Empower your motherhood journey with expert advice and support tailored to your baby's needs."
      imageSource={require("@/assets/images/mother.png")}
      currentPage={1}
      totalPages={4}
      onNext={handleNext}
      onSkip={handleSkip}
    />
  );
}
