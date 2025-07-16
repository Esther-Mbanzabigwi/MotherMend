import { StyleSheet } from "react-native";
import { Colors } from "./Colors";

export const SharedStyles = StyleSheet.create({
  // Container styles
  container: {
    flex: 1,
  },
  contentContainer: {
    padding: 20,
  },
  
  // Card styles
  card: {
    backgroundColor: Colors.light.card,
    borderRadius: 16,
    padding: 16,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  
  // Typography
  title: {
    fontSize: 24,
    fontWeight: "bold",
    color: Colors.light.primary,
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: Colors.light.gray[500],
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "600",
    color: Colors.light.text,
    marginBottom: 8,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: Colors.light.gray[500],
    marginBottom: 16,
  },
  
  // Status badges
  badge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  badgeSuccess: {
    backgroundColor: "#DCFCE7",
  },
  badgeText: {
    fontWeight: "600",
  },
  badgeTextSuccess: {
    color: Colors.light.success,
  },
  
  // Grid layouts
  grid: {
    flexDirection: "row",
    gap: 16,
    marginBottom: 32,
  },
  
  // Activity items
  activityItem: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 16,
  },
  activityIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.light.blue[50],
    alignItems: "center",
    justifyContent: "center",
    marginRight: 12,
  },
  
  // Form styles
  formGroup: {
    marginBottom: 16,
  },
  label: {
    fontSize: 16,
    fontWeight: "600",
    color: Colors.light.text,
    marginBottom: 8,
  },
  errorText: {
    color: Colors.light.danger,
    fontSize: 14,
    marginTop: 4,
  },
}); 