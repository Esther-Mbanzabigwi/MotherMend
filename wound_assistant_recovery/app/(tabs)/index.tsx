import { Link } from "expo-router";
import { StyleSheet, Text, View } from "react-native";
import AppLayout from "../../components/AppLayout";
import Button from "../../components/ui/Button";
import { SharedStyles } from "../../constants/SharedStyles";
import { useAuthContext } from "../../context/authcontext";

export default function OverviewScreen() {
  const { user, logout } = useAuthContext();

  return (
    <AppLayout>
      <View style={styles.header}>
        <Text style={SharedStyles.title}>
          Welcome back, {user?.username || "User"}!
        </Text>
        <Text style={SharedStyles.subtitle}>
          Track your recovery progress and stay healthy
        </Text>
      </View>

      <View style={styles.content}>
        <View style={[SharedStyles.card, styles.statusCard]}>
          <View style={styles.statusHeader}>
            <Text style={SharedStyles.sectionTitle}>Recovery Status</Text>
            <View style={[SharedStyles.badge, SharedStyles.badgeSuccess]}>
              <Text style={[SharedStyles.badgeText, SharedStyles.badgeTextSuccess]}>Healthy</Text>
            </View>
          </View>
          <Text style={styles.lastChecked}>Last checked 2 hours ago</Text>
        </View>

        <View style={SharedStyles.grid}>
          <View style={[SharedStyles.card, styles.metricCard]}>
            <View style={SharedStyles.activityIcon}>
              <Text>🗓️</Text>
            </View>
            <Text style={styles.metricValue}>12</Text>
            <Text style={styles.metricLabel}>Days Since Surgery</Text>
            <Text style={styles.metricStatus}>Recovery on track</Text>
          </View>

          <View style={[SharedStyles.card, styles.metricCard]}>
            <View style={SharedStyles.activityIcon}>
              <Text>📸</Text>
            </View>
            <Text style={styles.metricValue}>8</Text>
            <Text style={styles.metricLabel}>Photos Taken</Text>
            <Text style={styles.metricStatus}>Last photo today</Text>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={SharedStyles.sectionTitle}>Quick Actions</Text>
          <Text style={SharedStyles.sectionSubtitle}>
            Common tasks for your recovery tracking
          </Text>

          <View style={styles.actionGrid}>
            <Link href="/capture" asChild>
              <Button
                title="Take New Photo"
                variant="primary"
                onPress={() => {}}
                style={styles.actionButton}
              />
            </Link>

            <Link href="/history" asChild>
              <Button
                title="View History"
                variant="secondary"
                onPress={() => {}}
                style={styles.actionButton}
              />
            </Link>

            <Link href="/hospitals" asChild>
              <Button
                title="Find Hospitals"
                variant="secondary"
                onPress={() => {}}
                style={styles.actionButton}
              />
            </Link>

            <Button
              title="Logout"
              variant="outline"
              style={styles.actionButton}
              onPress={logout}
            />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={SharedStyles.sectionTitle}>Recent Activity</Text>
          <View style={styles.activityList}>
            <View style={SharedStyles.activityItem}>
              <View style={SharedStyles.activityIcon}>
                <Text>✅</Text>
              </View>
              <View style={styles.activityContent}>
                <Text style={styles.activityTitle}>
                  Wound assessment - Healthy
                </Text>
                <Text style={styles.activityTime}>2 hours ago</Text>
              </View>
            </View>

            <View style={SharedStyles.activityItem}>
              <View style={SharedStyles.activityIcon}>
                <Text>📸</Text>
              </View>
              <View style={styles.activityContent}>
                <Text style={styles.activityTitle}>Photo captured</Text>
                <Text style={styles.activityTime}>2 hours ago</Text>
              </View>
            </View>

            <View style={SharedStyles.activityItem}>
              <View style={SharedStyles.activityIcon}>
                <Text>✅</Text>
              </View>
              <View style={styles.activityContent}>
                <Text style={styles.activityTitle}>
                  Wound assessment - Healthy
                </Text>
                <Text style={styles.activityTime}>1 day ago</Text>
              </View>
            </View>
          </View>
        </View>
      </View>
    </AppLayout>
  );
}

const styles = StyleSheet.create({
  header: {
    marginBottom: 16,
  },
  content: {
    flex: 1,
  },
  statusCard: {
    marginBottom: 24,
  },
  statusHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  lastChecked: {
    fontSize: 14,
    color: "#666",
  },
  metricCard: {
    flex: 1,
    backgroundColor: "#F5F7FF",
    alignItems: "center",
  },
  metricValue: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#1849D7",
    marginBottom: 4,
  },
  metricLabel: {
    fontSize: 14,
    color: "#666",
    marginBottom: 4,
    textAlign: "center",
  },
  metricStatus: {
    fontSize: 12,
    color: "#22C55E",
    fontWeight: "500",
  },
  section: {
    marginBottom: 32,
  },
  actionGrid: {
    gap: 12,
  },
  actionButton: {
    marginBottom: 0,
  },
  activityList: {
    gap: 16,
  },
  activityContent: {
    flex: 1,
  },
  activityTitle: {
    fontSize: 16,
    color: "#333",
    marginBottom: 4,
  },
  activityTime: {
    fontSize: 14,
    color: "#666",
  },
});
