import { Ionicons } from '@expo/vector-icons';
import { Link } from 'expo-router';
import { Image, ScrollView, StyleSheet, Text, View } from 'react-native';
import Button from '../../components/ui/Button';
import { Colors } from '../../constants/Colors';

export default function LandingScreen() {
  return (
    <View style={styles.background}>
      {/* Static Header */}
      <View style={styles.staticHeader}>
        <View style={styles.logoContainer}>
          <View style={styles.logoInnerCircle}>
            <Image
              source={require('../../assets/images/baby_hand.png')}
              style={styles.logoImage}
              resizeMode="contain"
            />
          </View>
        </View>
        <Text style={styles.title}>MotherMend</Text>
        <Text style={styles.subtitle}>Your trusted companion for C-section recovery</Text>
      </View>
      {/* Scrollable Content */}
      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.featuresSection}>
          <View style={styles.featureCard}>
            <Ionicons name="pulse" size={22} color={Colors.light.primary} style={styles.featureIcon} />
            <Text style={styles.featureText}>Track your healing with daily wound monitoring</Text>
          </View>
          <View style={styles.featureCard}>
            <Ionicons name="notifications" size={22} color={Colors.light.primary} style={styles.featureIcon} />
            <Text style={styles.featureText}>Get instant health alerts and expert tips</Text>
          </View>
          <View style={styles.featureCard}>
            <Ionicons name="navigate" size={22} color={Colors.light.primary} style={styles.featureIcon} />
            <Text style={styles.featureText}>Find nearby hospitals and support</Text>
          </View>
          <View style={styles.featureCard}>
            <Ionicons name="happy" size={22} color={Colors.light.primary} style={styles.featureIcon} />
            <Text style={styles.featureText}>Feel confident, supported, and informed</Text>
          </View>
        </View>

        <Text style={styles.energyTagline}>Ready to take charge of your recovery?</Text>
        <View style={styles.actions}>
          <Link href="/(auth)/register" asChild>
            <Button
              variant="primary"
              style={styles.energyButton}
              title={
                <View style={styles.buttonContent}>
                  <Ionicons name="rocket" size={22} color="#fff" style={{ marginRight: 8 }} />
                  <Text style={styles.energyButtonText}>Get Started</Text>
                </View>
              }
            />
          </Link>
          <Link href="/(auth)/login" asChild>
            <Button variant="secondary" style={styles.button} title="Login" />
          </Link>
        </View>
      </ScrollView>
    </View>
  );
}

const HEADER_HEIGHT = 210;

const styles = StyleSheet.create({
  background: {
    flex: 1,
    backgroundColor: Colors.light.background,
    justifyContent: 'flex-start',
    alignItems: 'center',
    paddingTop: 0,
  },
  staticHeader: {
    width: '100%',
    backgroundColor: Colors.light.primary,
    borderBottomLeftRadius: 32,
    borderBottomRightRadius: 32,
    alignItems: 'center',
    paddingBottom: 18,
    paddingTop: 40,
    shadowColor: Colors.light.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.10,
    shadowRadius: 12,
    elevation: 6,
    zIndex: 10,
  },
  scrollContent: {
    alignItems: 'center',
    paddingBottom: 40,
    paddingTop: 18,
    minHeight: 400,
  },
  logoContainer: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: Colors.light.blue[500],
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
    shadowColor: Colors.light.background,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.12,
    shadowRadius: 8,
    elevation: 4,
  },
  logoInnerCircle: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  logoImage: {
    width: '80%',
    height: '80%',
    borderRadius: 40,
  },
  title: {
    fontSize: 34,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 4,
    letterSpacing: 1,
  },
  subtitle: {
    fontSize: 18,
    color: Colors.light.blue[50],
    textAlign: 'center',
    marginBottom: 6,
    fontWeight: '500',
  },
  featuresSection: {
    width: '100%',
    maxWidth: 400,
    marginBottom: 28,
    paddingHorizontal: 24,
  },
  featureCard: {
    backgroundColor: '#fff',
    borderRadius: 14,
    paddingVertical: 16,
    paddingHorizontal: 18,
    marginBottom: 14,
    alignItems: 'center',
    flexDirection: 'row',
    shadowColor: Colors.light.primary,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 2,
  },
  featureIcon: {
    marginRight: 12,
  },
  featureText: {
    fontSize: 16,
    color: Colors.light.primary,
    textAlign: 'left',
    fontWeight: '500',
    flex: 1,
  },
  energyTagline: {
    fontSize: 17,
    color: Colors.light.primary,
    fontWeight: '600',
    textAlign: 'center',
    marginBottom: 18,
    marginTop: 8,
    letterSpacing: 0.5,
  },
  actions: {
    width: '100%',
    maxWidth: 400,
    paddingHorizontal: 24,
    marginTop: 8,
  },
  energyButton: {
    marginBottom: 14,
    borderRadius: 18,
    backgroundColor: Colors.light.primary,
    shadowColor: Colors.light.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.18,
    shadowRadius: 12,
    elevation: 6,
    paddingVertical: 18,
    paddingHorizontal: 12,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
  },
  buttonContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  energyButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    letterSpacing: 0.5,
  },
  button: {
    marginBottom: 10,
    borderRadius: 14,
  },
});
