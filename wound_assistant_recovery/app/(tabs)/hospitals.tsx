import { Ionicons } from "@expo/vector-icons";
import {
  BottomSheetModal,
  BottomSheetModalProvider,
} from "@gorhom/bottom-sheet";
import * as Linking from "expo-linking";
import debounce from "lodash.debounce";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  StyleSheet,
  Text,
  View,
} from "react-native";
import MapView, { Marker, PROVIDER_GOOGLE } from "react-native-maps";
import { HospitalAPI } from "../../api/hospitalApi";
import AppLayout from "../../components/AppLayout";
import HospitalBottomSheet from "../../components/HospitalBottomSheet";
import Button from "../../components/ui/Button";
import Input from "../../components/ui/Input";
import { Colors } from "../../constants/Colors";
import { SharedStyles } from "../../constants/SharedStyles";
import { Hospital, Location } from "../../types/hospitalType";

export default function HospitalsScreen() {
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [userLocation, setUserLocation] = useState<Location | null>(null);
  const [selectedHospital, setSelectedHospital] = useState<Hospital | null>(
    null
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const bottomSheetRef = useRef<BottomSheetModal>(null);

  // Debounced search function
  const debouncedSearch = useRef(
    debounce(async (query: string, location: Location) => {
      if (!query.trim() || !location) return;

      try {
        setSearching(true);
        setError("");
        
        const results = await HospitalAPI.searchHospitals(query, location);
        setHospitals(results);
        
        console.log("Search results:", results.map(h => `${h.name} - ${h.address} (${h.distance?.toFixed(1)} miles)`));
        
        if (results.length === 0) {
          setError(`No hospitals found matching "${query}". Try a different search term.`);
        }
      } catch (error) {
        console.error("Error searching hospitals:", error);
        setError("Failed to search hospitals. Please try again.");
      } finally {
        setSearching(false);
      }
    }, 800) // 800ms delay to prevent excessive API calls
  ).current;

  // Get user's current location and nearby hospitals
  const getCurrentLocation = useCallback(async () => {
    try {
      setLoading(true);
      setError("");
      setSearchQuery(""); // Clear search when refreshing location
      
      const location = await HospitalAPI.getCurrentLocation();
      setUserLocation(location);

      const nearbyHospitals = await HospitalAPI.getNearbyHospitals(location, 100); // Increased radius to 100 miles
      setHospitals(nearbyHospitals);
      
      console.log("Found hospitals:", nearbyHospitals.map(h => `${h.name} - ${h.address} (${h.distance?.toFixed(1)} miles)`));
      
      if (nearbyHospitals.length === 0) {
        setError("No hospitals found nearby. Try searching for a specific hospital or expanding the search radius.");
      }
    } catch (error) {
      console.error("Error getting location:", error);
      setError("Unable to get your location. Please check your location permissions and try again.");
    } finally {
      setLoading(false);
    }
  }, []);

  // Manual search function (for search button)
  const searchHospitals = useCallback(async () => {
    if (!searchQuery.trim() || !userLocation) return;

    try {
      setSearching(true);
      setError("");
      
      const results = await HospitalAPI.searchHospitals(searchQuery, userLocation);
      setHospitals(results);
      
      console.log("Search results:", results.map(h => `${h.name} - ${h.address} (${h.distance?.toFixed(1)} miles)`));
      
      if (results.length === 0) {
        setError(`No hospitals found matching "${searchQuery}". Try a different search term.`);
      }
    } catch (error) {
      console.error("Error searching hospitals:", error);
      setError("Failed to search hospitals. Please try again.");
    } finally {
      setSearching(false);
    }
  }, [searchQuery, userLocation]);

  // Handle search input change with debouncing
  const handleSearchChange = useCallback((text: string) => {
    setSearchQuery(text);
    
    if (!text.trim()) {
      // Clear search and reset to nearby hospitals
      if (userLocation) {
        getCurrentLocation();
      }
      return;
    }

    // Debounce the search to prevent excessive API calls
    if (userLocation) {
      debouncedSearch(text, userLocation);
    }
  }, [userLocation, getCurrentLocation, debouncedSearch]);

  // Cleanup debounced function on unmount
  useEffect(() => {
    return () => {
      debouncedSearch.cancel();
    };
  }, [debouncedSearch]);

  // Handle marker press
  const handleMarkerPress = useCallback((hospital: Hospital) => {
    setSelectedHospital(hospital);
    bottomSheetRef.current?.present();
  }, []);

  // Handle get directions
  const handleGetDirections = useCallback(async (hospital: Hospital) => {
    try {
      const url = `https://www.google.com/maps/dir/?api=1&destination=${hospital.coordinates.latitude},${hospital.coordinates.longitude}`;
      const supported = await Linking.canOpenURL(url);

      if (supported) {
        await Linking.openURL(url);
      } else {
        Alert.alert("Error", "Unable to open maps application");
      }
    } catch (error) {
      console.error("Error opening directions:", error);
      Alert.alert("Error", "Unable to open directions");
    }
  }, []);

  // Handle call hospital
  const handleCallHospital = useCallback(async (phone: string) => {
    try {
      const url = `tel:${phone}`;
      const supported = await Linking.canOpenURL(url);

      if (supported) {
        await Linking.openURL(url);
      } else {
        Alert.alert("Error", "Unable to make phone call");
      }
    } catch (error) {
      console.error("Error calling hospital:", error);
      Alert.alert("Error", "Unable to make phone call");
    }
  }, []);

  // On mount: fetch current location
  useEffect(() => {
    getCurrentLocation();
  }, [getCurrentLocation]);

  return (
    <BottomSheetModalProvider>
      <AppLayout>
        <View style={styles.header}>
          <Text style={SharedStyles.title}>Find Hospitals</Text>
          <Text style={SharedStyles.subtitle}>
            Search for medical facilities or find nearby hospitals
          </Text>
        </View>

        {/* Search Section */}
        <View style={styles.searchSection}>
          <Input
            placeholder="Search hospitals by name or specialty..."
            icon="search"
            style={styles.searchInput}
            value={searchQuery}
            onChangeText={handleSearchChange}
            onSubmitEditing={searchHospitals}
          />
          <Button
            title={searching ? "Searching..." : "Search"}
            variant="primary"
            onPress={searchHospitals}
            style={styles.searchButton}
            disabled={searching || !searchQuery.trim()}
          />
        </View>

        {/* Location Section */}
        <View style={styles.locationSection}>
          <Button
            title={loading ? "Loading..." : "Use My Location"}
            variant="secondary"
            onPress={getCurrentLocation}
            style={styles.locationButton}
            disabled={loading}
          />
        </View>

        {userLocation && (
          <View style={styles.locationInfo}>
            <Text style={styles.locationText}>Current Location</Text>
            <Text style={styles.address}>{userLocation.address}</Text>
            <Text style={styles.dataSource}>Using real hospital data from Google Places API</Text>
          </View>
        )}

        {loading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={Colors.light.primary} />
            <Text style={styles.loadingText}>Finding real hospitals near you...</Text>
          </View>
        )}

        {searching && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="small" color={Colors.light.primary} />
            <Text style={styles.loadingText}>Searching hospitals...</Text>
          </View>
        )}

        {error ? (
          <View style={styles.errorContainer}>
            <Text style={SharedStyles.errorText}>{error}</Text>
            <Button
              title="Try Again"
              variant="secondary"
              onPress={searchQuery.trim() ? searchHospitals : getCurrentLocation}
              style={styles.retryButton}
            />
          </View>
        ) : null}

        {/* Hospital Count */}
        {hospitals.length > 0 && (
          <View style={styles.hospitalCount}>
            <Text style={styles.countText}>
              Found {hospitals.length} hospital{hospitals.length !== 1 ? 's' : ''}
            </Text>
          </View>
        )}

        {/* Map Section - Made Larger */}
        <View style={styles.mapContainer}>
          {userLocation ? (
            <MapView
              provider={PROVIDER_GOOGLE}
              style={styles.map}
              initialRegion={{
                latitude: userLocation.latitude,
                longitude: userLocation.longitude,
                latitudeDelta: 0.05, // Increased for better view
                longitudeDelta: 0.05,
              }}
              showsUserLocation={true}
              showsMyLocationButton={true}
              showsCompass={true}
              showsScale={true}
            >
              <Marker
                coordinate={{
                  latitude: userLocation.latitude,
                  longitude: userLocation.longitude,
                }}
                title="Your Location"
                description="You are here"
                pinColor={Colors.light.primary}
              />

              {hospitals.map((hospital) => (
                <Marker
                  key={hospital.id}
                  coordinate={{
                    latitude: hospital.coordinates.latitude,
                    longitude: hospital.coordinates.longitude,
                  }}
                  title={hospital.name}
                  description={`${hospital.address} • ${hospital.distance?.toFixed(1)} miles away`}
                  onPress={() => handleMarkerPress(hospital)}
                >
                  <View style={styles.markerContainer}>
                    <View style={styles.markerIcon}>
                      <Ionicons name="medical" size={16} color="#fff" />
                    </View>
                  </View>
                </Marker>
              ))}
            </MapView>
          ) : (
            <View style={styles.mapPlaceholder}>
              {loading ? (
                <ActivityIndicator size="large" color={Colors.light.primary} />
              ) : (
                <Text style={styles.mapPlaceholderText}>Loading map...</Text>
              )}
            </View>
          )}
        </View>

        <HospitalBottomSheet
          ref={bottomSheetRef}
          hospital={selectedHospital}
          onGetDirections={handleGetDirections}
          onCall={handleCallHospital}
        />
      </AppLayout>
    </BottomSheetModalProvider>
  );
}

const styles = StyleSheet.create({
  header: {
    marginBottom: 20,
  },
  searchSection: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 16,
  },
  searchInput: {
    flex: 1,
  },
  searchButton: {
    minWidth: 80,
  },
  locationSection: {
    marginBottom: 16,
  },
  locationButton: {
    width: "100%",
  },
  locationInfo: {
    backgroundColor: Colors.light.blue[50],
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  locationText: {
    fontSize: 14,
    color: Colors.light.gray[500],
    marginBottom: 4,
  },
  address: {
    fontSize: 16,
    color: Colors.light.text,
    fontWeight: "500",
  },
  dataSource: {
    fontSize: 12,
    color: Colors.light.gray[400],
    marginTop: 4,
    fontStyle: "italic",
  },
  loadingContainer: {
    alignItems: "center",
    marginBottom: 16,
  },
  loadingText: {
    marginTop: 8,
    fontSize: 14,
    color: Colors.light.gray[500],
  },
  errorContainer: {
    marginBottom: 16,
  },
  retryButton: {
    marginTop: 8,
  },
  hospitalCount: {
    marginBottom: 12,
  },
  countText: {
    fontSize: 14,
    color: Colors.light.gray[600],
    fontWeight: "500",
  },
  mapContainer: {
    flex: 1,
    borderRadius: 12,
    overflow: "hidden",
    marginBottom: 16,
    minHeight: 400, // Ensure minimum height for better visibility
  },
  map: {
    width: "100%",
    height: "100%",
  },
  mapPlaceholder: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: Colors.light.gray[100],
    minHeight: 400,
  },
  mapPlaceholderText: {
    fontSize: 16,
    color: Colors.light.gray[500],
  },
  markerContainer: {
    alignItems: "center",
  },
  markerIcon: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: Colors.light.primary,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 2,
    borderColor: "#fff",
  },
});
