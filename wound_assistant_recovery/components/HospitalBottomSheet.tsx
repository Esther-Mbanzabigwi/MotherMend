import { Ionicons } from "@expo/vector-icons";
import { BottomSheetModal, BottomSheetView } from "@gorhom/bottom-sheet";
import { forwardRef, useCallback, useMemo } from "react";
import { StyleSheet, Text, View } from "react-native";
import { Colors } from "../constants/Colors";
import { Hospital } from "../types/hospitalType";
import Button from "./ui/Button";

interface HospitalBottomSheetProps {
  hospital: Hospital | null;
  onGetDirections: (hospital: Hospital) => void;
  onCall: (phone: string) => void;
}

const HospitalBottomSheet = forwardRef<
  BottomSheetModal,
  HospitalBottomSheetProps
>(({ hospital, onGetDirections, onCall }, ref) => {
  const snapPoints = useMemo(() => ["50%", "75%"], []);

  const handleGetDirections = useCallback(() => {
    if (hospital) {
      onGetDirections(hospital);
    }
  }, [hospital, onGetDirections]);

  const handleCall = useCallback(() => {
    if (hospital) {
      onCall(hospital.phone);
    }
  }, [hospital, onCall]);

  if (!hospital) return null;

  return (
    <BottomSheetModal
      ref={ref}
      index={0}
      snapPoints={snapPoints}
      backgroundStyle={styles.bottomSheetBackground}
      handleIndicatorStyle={styles.handleIndicator}
    >
      <BottomSheetView style={styles.content}>
        <View style={styles.header}>
          <View style={styles.hospitalInfo}>
            <Text style={styles.hospitalName}>{hospital.name}</Text>
            <Text style={styles.distance}>
              {hospital.distance
                ? `${hospital.distance.toFixed(1)} miles away`
                : ""}
            </Text>
          </View>
          <View style={styles.iconContainer}>
            <Ionicons name="medical" size={24} color={Colors.light.primary} />
          </View>
        </View>

        <View style={styles.details}>
          <View style={styles.detailRow}>
            <Ionicons
              name="location"
              size={16}
              color={Colors.light.gray[500]}
            />
            <Text style={styles.detailText}>{hospital.address}</Text>
          </View>

          <View style={styles.detailRow}>
            <Ionicons name="call" size={16} color={Colors.light.gray[500]} />
            <Text style={styles.detailText}>{hospital.phone}</Text>
          </View>

          <View style={styles.detailRow}>
            <Ionicons name="time" size={16} color={Colors.light.gray[500]} />
            <Text style={styles.detailText}>{hospital.hours}</Text>
          </View>

          <View style={styles.detailRow}>
            <Ionicons name="timer" size={16} color={Colors.light.gray[500]} />
            <Text style={styles.detailText}>
              Wait Time: {hospital.waitTime}
            </Text>
          </View>
        </View>

        <View style={styles.specialties}>
          <Text style={styles.specialtiesTitle}>Specialties:</Text>
          <View style={styles.specialtyTags}>
            {hospital.specialties.map((specialty, index) => (
              <Text key={index} style={styles.specialtyTag}>
                {specialty}
              </Text>
            ))}
          </View>
        </View>

        <View style={styles.actions}>
          <Button
            title="Get Directions"
            variant="primary"
            onPress={handleGetDirections}
            style={styles.actionButton}
          />
          <Button
            title="Call"
            variant="secondary"
            onPress={handleCall}
            style={styles.actionButton}
          />
        </View>
      </BottomSheetView>
    </BottomSheetModal>
  );
});

HospitalBottomSheet.displayName = "HospitalBottomSheet";

const styles = StyleSheet.create({
  bottomSheetBackground: {
    backgroundColor: Colors.light.background,
  },
  handleIndicator: {
    backgroundColor: Colors.light.gray[300],
  },
  content: {
    flex: 1,
    padding: 20,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 20,
  },
  hospitalInfo: {
    flex: 1,
  },
  hospitalName: {
    fontSize: 20,
    fontWeight: "bold",
    color: Colors.light.primary,
    marginBottom: 4,
  },
  distance: {
    fontSize: 14,
    color: Colors.light.gray[500],
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: Colors.light.blue[50],
    justifyContent: "center",
    alignItems: "center",
  },
  details: {
    marginBottom: 20,
  },
  detailRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 12,
  },
  detailText: {
    fontSize: 16,
    color: Colors.light.text,
    marginLeft: 8,
    flex: 1,
  },
  specialties: {
    marginBottom: 24,
  },
  specialtiesTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: Colors.light.text,
    marginBottom: 8,
  },
  specialtyTags: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
  },
  specialtyTag: {
    backgroundColor: Colors.light.blue[50],
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    fontSize: 14,
    color: Colors.light.primary,
  },
  actions: {
    flexDirection: "row",
    gap: 12,
  },
  actionButton: {
    flex: 1,
  },
});

export default HospitalBottomSheet;
