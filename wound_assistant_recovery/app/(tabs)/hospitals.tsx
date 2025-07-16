import { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import AppLayout from '../../components/AppLayout';
import Button from '../../components/ui/Button';
import Input from '../../components/ui/Input';
import { Colors } from '../../constants/Colors';
import { SharedStyles } from '../../constants/SharedStyles';

export default function HospitalsScreen() {
  const [searchQuery, setSearchQuery] = useState('');

  return (
    <AppLayout>
      <View style={styles.header}>
        <Text style={SharedStyles.title}>Find Nearby Hospitals</Text>
        <Text style={SharedStyles.subtitle}>
          Locate medical facilities near you for immediate care
        </Text>
      </View>

      <View style={styles.searchContainer}>
        <Input
          placeholder="Search hospitals"
          icon="search"
          style={styles.searchInput}
          value={searchQuery}
          onChangeText={setSearchQuery}
        />
        <Button
          title="Use My Location"
          variant="primary"
          onPress={() => {}}
          style={styles.locationButton}
        />
      </View>

      <View style={styles.locationInfo}>
        <Text style={styles.locationText}>Current Location</Text>
        <Text style={styles.address}>Downtown Area, Main Street</Text>
      </View>

      <View style={styles.hospitalList}>
        <View style={[SharedStyles.card, styles.hospitalCard]}>
          <View style={styles.hospitalInfo}>
            <Text style={styles.hospitalName}>City General Hospital</Text>
            <Text style={styles.distance}>0.8 miles away</Text>
          </View>
          
          <Text style={styles.hospitalAddress}>123 Main Street, Downtown</Text>
          <Text style={styles.phone}>(555) 123-4567</Text>
          <Text style={styles.hours}>24/7</Text>

          <Text style={styles.waitTime}>Wait Time: ~15 mins</Text>

          <View style={styles.specialties}>
            <Text style={styles.specialtyTag}>Emergency Care</Text>
            <Text style={styles.specialtyTag}>Maternity</Text>
            <Text style={styles.specialtyTag}>Surgery</Text>
          </View>

          <View style={styles.actions}>
            <Button
              title="Get Directions"
              variant="primary"
              onPress={() => {}}
              style={styles.actionButton}
            />
            <Button
              title="Call"
              variant="secondary"
              onPress={() => {}}
              style={styles.actionButton}
            />
          </View>
        </View>

        <View style={[SharedStyles.card, styles.hospitalCard]}>
          <View style={styles.hospitalInfo}>
            <Text style={styles.hospitalName}>Women's Health Center</Text>
            <Text style={styles.distance}>1.2 miles away</Text>
          </View>
          
          <Text style={styles.hospitalAddress}>456 Oak Avenue, Midtown</Text>
          <Text style={styles.phone}>(555) 987-6543</Text>
          <Text style={styles.hours}>6 AM - 10 PM</Text>

          <Text style={styles.waitTime}>Wait Time: ~30 mins</Text>

          <View style={styles.specialties}>
            <Text style={styles.specialtyTag}>Women's Health</Text>
            <Text style={styles.specialtyTag}>Postpartum Care</Text>
            <Text style={styles.specialtyTag}>OB/GYN</Text>
          </View>

          <View style={styles.actions}>
            <Button
              title="Get Directions"
              variant="primary"
              onPress={() => {}}
              style={styles.actionButton}
            />
            <Button
              title="Call"
              variant="secondary"
              onPress={() => {}}
              style={styles.actionButton}
            />
          </View>
        </View>
      </View>
    </AppLayout>
  );
}

const styles = StyleSheet.create({
  header: {
    marginBottom: 24,
  },
  searchContainer: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  searchInput: {
    flex: 1,
  },
  locationButton: {
    minWidth: 140,
  },
  locationInfo: {
    backgroundColor: Colors.light.blue[50],
    padding: 12,
    borderRadius: 8,
    marginBottom: 24,
  },
  locationText: {
    fontSize: 14,
    color: Colors.light.gray[500],
    marginBottom: 4,
  },
  address: {
    fontSize: 16,
    color: Colors.light.text,
    fontWeight: '500',
  },
  hospitalList: {
    flex: 1,
  },
  hospitalCard: {
    marginBottom: 16,
  },
  hospitalInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  hospitalName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: Colors.light.primary,
  },
  distance: {
    fontSize: 14,
    color: Colors.light.gray[500],
  },
  hospitalAddress: {
    fontSize: 16,
    color: Colors.light.text,
    marginBottom: 4,
  },
  phone: {
    fontSize: 16,
    color: Colors.light.text,
    marginBottom: 4,
  },
  hours: {
    fontSize: 14,
    color: Colors.light.gray[500],
    marginBottom: 8,
  },
  waitTime: {
    fontSize: 14,
    color: Colors.light.gray[500],
    marginBottom: 12,
  },
  specialties: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 16,
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
    flexDirection: 'row',
    gap: 12,
  },
  actionButton: {
    flex: 1,
  },
}); 