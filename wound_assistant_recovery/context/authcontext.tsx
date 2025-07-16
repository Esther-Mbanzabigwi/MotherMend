import { UserHandler } from "@/api/userapi";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { router } from "expo-router";
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { AuthContextType } from "../types/context.type";
import type { ILogInUser, IRegisterUser, IUser } from "../types/usertype";

const AuthContext = createContext<AuthContextType>({
  user: null,
  token: null,
  isInitialized: false,
  hasSeenOnboarding: false,
  login: async () => {
    throw new Error("Auth context not initialized");
  },
  logout: async () => {},
  register: async () => {},
  completeOnboarding: async () => {},
});

export function useAuthContext() {
  return useContext(AuthContext);
}

type ProviderProps = {
  children: ReactNode;
};

export default function AuthContextProvider({ children }: ProviderProps) {
  const [user, setUser] = useState<IUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [hasSeenOnboarding, setHasSeenOnboarding] = useState(false);

  useEffect(() => {
    const loadStoredAuth = async () => {
      try {
        const [token, currentUser, onboardingStatus] = await Promise.all([
          AsyncStorage.getItem("token"),
          AsyncStorage.getItem("user"),
          AsyncStorage.getItem("hasSeenOnboarding"),
        ]);

        if (token && currentUser) {
          setToken(token);
          setUser(JSON.parse(currentUser));
        } else {
          setUser(null);
          setToken(null);
          await Promise.all([
            AsyncStorage.removeItem("token"),
            AsyncStorage.removeItem("user"),
          ]);
        }

        setHasSeenOnboarding(onboardingStatus === "true");
      } catch (error) {
        console.error("Error loading stored auth:", error);
        setUser(null);
        setToken(null);
      } finally {
        setIsInitialized(true);
      }
    };

    loadStoredAuth();
  }, []);

  async function login(userInfo: ILogInUser) {
    try {
      const response = await UserHandler.logIn(userInfo);
      setToken(response.jwt);
      await AsyncStorage.setItem("token", response.jwt);
      setUser(response.user);
      await Promise.all([
        AsyncStorage.setItem("user", JSON.stringify(response.user)),
      ]);
      return response.user;
    } catch (error) {
      console.error("Error during login:", error);
      throw new Error("An error occurred. Please try again later");
    }
  }

  async function logout() {
    setUser(null);
    setToken(null);
    await Promise.all([
      AsyncStorage.removeItem("token"),
      AsyncStorage.removeItem("user"),
    ]);
    router.replace("/(auth)");
  }

  async function register(userInfo: IRegisterUser) {
    try {
      const response = await UserHandler.register(userInfo);
      setUser(response.user);
      setToken(response.jwt);
      await Promise.all([
        AsyncStorage.setItem("token", response.jwt),
        AsyncStorage.setItem("user", JSON.stringify(response.user)),
      ]);
    } catch (error) {
      console.error("Error during registration:", error);
      throw new Error("An error occurred. Please try again later!");
    }
  }

  async function completeOnboarding() {
    setHasSeenOnboarding(true);
    await AsyncStorage.setItem("hasSeenOnboarding", "true");
  }

  const values = useMemo(() => {
    return {
      user,
      token,
      isInitialized,
      hasSeenOnboarding,
      login,
      logout,
      register,
      completeOnboarding,
    };
  }, [user, token, isInitialized, hasSeenOnboarding]);
  return <AuthContext.Provider value={values}>{children}</AuthContext.Provider>;
}
