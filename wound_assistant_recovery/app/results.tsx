import { useLocalSearchParams, useRouter } from 'expo-router';
import { Image, StyleSheet, Text, View } from 'react-native';
import { WoundPrediction } from '../api';
import AppLayout from '../components/AppLayout';
import Button from '../components/ui/Button';
import { Colors } from '../constants/Colors';
import { SharedStyles } from '../constants/SharedStyles';

export default function ResultsScreen() {
  const router = useRouter();
  const { prediction, imageUri } = useLocalSearchParams<{
    prediction: string;
    imageUri: string;
  }>();

  const predictionData: WoundPrediction = JSON.parse(prediction);

  const getUrgencyColor = (level: string) => {
    switch (level.toUpperCase()) {
      case 'HIGH':
        return Colors.light.danger;
      case 'MEDIUM':
        return Colors.light.warning;
      case 'LOW':
        return Colors.light.success;
      default:
        return Colors.light.gray[500];
    }
  };

  return (
    <AppLayout>
      <View style={styles.header}>
        <Text style={SharedStyles.title}>Analysis Results</Text>
        <Text style={SharedStyles.subtitle}>
          Here's what we found from analyzing your wound
        </Text>
      </View>

      <View style={styles.imageContainer}>
        <Image source={{ uri: imageUri }} style={styles.image} />
      </View>

      <View style={[SharedStyles.card, styles.resultCard]}>
        <View style={styles.resultHeader}>
          <Text style={styles.resultTitle}>Classification</Text>
          <Text style={styles.confidence}>
            {(predictionData.confidence * 100).toFixed(1)}% confidence
          </Text>
        </View>
        <Text style={styles.classification}>
          {predictionData.predicted_class}
        </Text>

        <View style={styles.urgencyContainer}>
          <Text style={styles.urgencyLabel}>Urgency Level:</Text>
          <View
            style={[
              styles.urgencyBadge,
              { backgroundColor: getUrgencyColor(predictionData.urgency_level) },
            ]}
          >
            <Text style={styles.urgencyText}>
              {predictionData.urgency_level}
            </Text>
          </View>
        </View>

        <View style={styles.hospitalContainer}>
          <Text style={styles.hospitalText}>
            {predictionData.requires_hospital
              ? '⚠️ Hospital visit recommended'
              : '✓ No immediate hospital visit needed'}
          </Text>
        </View>

        <View style={styles.recommendationsContainer}>
          <Text style={styles.recommendationsTitle}>Recommendations:</Text>
          {predictionData.recommendations.map((rec, index) => (
            <View key={index} style={styles.recommendationItem}>
              <Text style={styles.recommendationBullet}>•</Text>
              <Text style={styles.recommendationText}>{rec}</Text>
            </View>
          ))}
        </View>
      </View>

      <View style={styles.actions}>
        <Button
          title="Take Another Photo"
          variant="secondary"
          onPress={() => router.back()}
          style={styles.button}
        />
        <Button
          title="View History"
          variant="primary"
          onPress={() => router.push('/history')}
          style={styles.button}
        />
      </View>
    </AppLayout>
  );
}

const styles = StyleSheet.create({
  header: {
    marginBottom: 24,
  },
  imageContainer: {
    backgroundColor: Colors.light.card,
    borderRadius: 16,
    overflow: 'hidden',
    aspectRatio: 4/3,
    marginBottom: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  image: {
    width: '100%',
    height: '100%',
  },
  resultCard: {
    marginBottom: 24,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.light.text,
  },
  confidence: {
    fontSize: 14,
    color: Colors.light.gray[500],
  },
  classification: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.light.primary,
    marginBottom: 16,
  },
  urgencyContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
  },
  urgencyLabel: {
    fontSize: 16,
    color: Colors.light.text,
  },
  urgencyBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  urgencyText: {
    color: 'white',
    fontWeight: '600',
    fontSize: 14,
  },
  hospitalContainer: {
    marginBottom: 16,
  },
  hospitalText: {
    fontSize: 16,
    color: Colors.light.text,
  },
  recommendationsContainer: {
    gap: 8,
  },
  recommendationsTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.light.text,
    marginBottom: 8,
  },
  recommendationItem: {
    flexDirection: 'row',
    gap: 8,
  },
  recommendationBullet: {
    color: Colors.light.primary,
    fontSize: 16,
  },
  recommendationText: {
    flex: 1,
    fontSize: 16,
    color: Colors.light.text,
    lineHeight: 24,
  },
  actions: {
    flexDirection: 'row',
    gap: 12,
  },
  button: {
    flex: 1,
  },
}); 