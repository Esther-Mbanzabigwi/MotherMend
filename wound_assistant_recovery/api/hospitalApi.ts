import * as Location from "expo-location";
import {
  Hospital,
  Location as LocationType,
  Route,
} from "../types/hospitalType";

// Google Places API configuration
const GOOGLE_PLACES_API_KEY = "AIzaSyBVkev2Nxc3OfWXY0bv1J0UrhXgmZsX0iw"; // Using the same key from AndroidManifest
const GOOGLE_PLACES_BASE_URL = "https://maps.googleapis.com/maps/api/place";

// Fallback mock data for when API is unavailable
const FALLBACK_HOSPITALS: Hospital[] = [
  // Kigali, Rwanda hospitals
  {
    id: "kigali_1",
    name: "King Faisal Hospital",
    address: "KG 2 Ave, Kigali, Rwanda",
    phone: "+250 788 303 030",
    hours: "24/7",
    waitTime: "~20 mins",
    specialties: ["Emergency Care", "Surgery", "Cardiology", "Oncology"],
    coordinates: {
      latitude: -1.9441,
      longitude: 30.0619,
    },
  },
  {
    id: "kigali_2",
    name: "Kigali University Teaching Hospital (CHUK)",
    address: "KN 4 Ave, Kigali, Rwanda",
    phone: "+250 788 303 030",
    hours: "24/7",
    waitTime: "~30 mins",
    specialties: ["Emergency Care", "Teaching Hospital", "General Medicine"],
    coordinates: {
      latitude: -1.9501,
      longitude: 30.0587,
    },
  },
  {
    id: "kigali_3",
    name: "Kibagabaga Hospital",
    address: "KG 17 St, Kibagabaga, Kigali, Rwanda",
    phone: "+250 788 303 030",
    hours: "7 AM - 10 PM",
    waitTime: "~15 mins",
    specialties: ["Family Medicine", "Pediatrics", "Emergency Care"],
    coordinates: {
      latitude: -1.9485,
      longitude: 30.1265,
    },
  },
  {
    id: "kigali_4",
    name: "Kacyiru Hospital",
    address: "KG 7 Ave, Kacyiru, Kigali, Rwanda",
    phone: "+250 788 303 030",
    hours: "6 AM - 11 PM",
    waitTime: "~25 mins",
    specialties: ["Women's Health", "Maternity", "Emergency Care"],
    coordinates: {
      latitude: -1.9439,
      longitude: 30.0594,
    },
  },
  {
    id: "kigali_5",
    name: "Kanombe Military Hospital",
    address: "KG 2 Ave, Kanombe, Kigali, Rwanda",
    phone: "+250 788 303 030",
    hours: "24/7",
    waitTime: "~10 mins",
    specialties: ["Military Hospital", "Emergency Care", "Trauma"],
    coordinates: {
      latitude: -1.9667,
      longitude: 30.1333,
    },
  },
  // General fallback hospitals (US coordinates)
  {
    id: "fallback_1",
    name: "City General Hospital",
    address: "123 Main Street, Downtown",
    phone: "(555) 123-4567",
    hours: "24/7",
    waitTime: "~15 mins",
    specialties: ["Emergency Care", "Maternity", "Surgery"],
    coordinates: {
      latitude: 37.7749,
      longitude: -122.4194,
    },
  },
  {
    id: "fallback_2",
    name: "Women's Health Center",
    address: "456 Oak Avenue, Midtown",
    phone: "(555) 987-6543",
    hours: "6 AM - 10 PM",
    waitTime: "~30 mins",
    specialties: ["Women's Health", "Postpartum Care", "OB/GYN"],
    coordinates: {
      latitude: 37.7849,
      longitude: -122.4094,
    },
  },
];

// Google Places API response types
interface GooglePlace {
  place_id: string;
  name: string;
  formatted_address: string;
  geometry: {
    location: {
      lat: number;
      lng: number;
    };
  };
  formatted_phone_number?: string;
  opening_hours?: {
    open_now: boolean;
    weekday_text?: string[];
  };
  types: string[];
  rating?: number;
  user_ratings_total?: number;
}

interface GooglePlacesResponse {
  results: GooglePlace[];
  status: string;
  next_page_token?: string;
}

export class HospitalAPI {
  // Get user's current location with proper validation
  static async getCurrentLocation(): Promise<LocationType> {
    try {
      // Check if location services are enabled
      const isEnabled = await Location.hasServicesEnabledAsync();
      if (!isEnabled) {
        throw new Error("Location services are disabled. Please enable location services in your device settings.");
      }

      // Request permissions with proper error handling
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== "granted") {
        throw new Error("Location permission denied. Please grant location permissions to find nearby hospitals.");
      }

      // Get current position with proper options
      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.Balanced, // Use balanced accuracy for faster response
      });

      // Validate location data
      if (!location.coords || !location.coords.latitude || !location.coords.longitude) {
        throw new Error("Unable to get your current location. Please try again.");
      }

      // Get address from coordinates with error handling
      let address = "Current Location";
      try {
        const addressResponse = await Location.reverseGeocodeAsync({
          latitude: location.coords.latitude,
          longitude: location.coords.longitude,
        });

        if (addressResponse && addressResponse.length > 0) {
          const addressData = addressResponse[0];
          const street = addressData.street || "";
          const city = addressData.city || "";
          const state = addressData.region || "";
          
          if (street && city) {
            address = `${street}, ${city}${state ? `, ${state}` : ""}`;
          } else if (city) {
            address = city;
          }
        }
      } catch (addressError) {
        console.warn("Could not get address, using default:", addressError);
        // Continue with default address
      }

      return {
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
        address,
      };
    } catch (error) {
      console.error("Error getting location:", error);
      
      // Provide user-friendly error messages
      if (error instanceof Error) {
        throw new Error(error.message);
      } else {
        throw new Error("Unable to get your location. Please check your location settings and try again.");
      }
    }
  }

  // Fetch hospitals from Google Places API
  static async fetchHospitalsFromAPI(
    latitude: number,
    longitude: number,
    radius: number = 50000, // 50km radius
    searchQuery?: string
  ): Promise<Hospital[]> {
    try {
      let url: string;
      
      if (searchQuery && searchQuery.trim()) {
        // Use text search for specific queries
        const encodedQuery = encodeURIComponent(searchQuery.trim());
        url = `${GOOGLE_PLACES_BASE_URL}/textsearch/json?query=${encodedQuery}&location=${latitude},${longitude}&radius=${radius}&type=hospital&key=${GOOGLE_PLACES_API_KEY}`;
      } else {
        // Use nearby search for general hospital search
        url = `${GOOGLE_PLACES_BASE_URL}/nearbysearch/json?location=${latitude},${longitude}&radius=${radius}&type=hospital&key=${GOOGLE_PLACES_API_KEY}`;
      }
      
      console.log("Fetching hospitals from:", url);
      const response = await fetch(url);
      const data: GooglePlacesResponse = await response.json();

      console.log("Google Places API response:", data.status, "Results:", data.results?.length || 0);

      if (data.status !== "OK" && data.status !== "ZERO_RESULTS") {
        console.warn("Google Places API error:", data.status);
        throw new Error(`API Error: ${data.status}`);
      }

      if (data.status === "ZERO_RESULTS" || !data.results) {
        return [];
      }

      // Transform Google Places data to our Hospital format
      const hospitals: Hospital[] = data.results.map((place, index) => {
        // Determine specialties based on place types
        const specialties = this.getSpecialtiesFromTypes(place.types);
        
        // Determine hours based on opening_hours
        const hours = place.opening_hours?.open_now ? "Open Now" : "Hours Vary";
        
        // Generate wait time estimate (mock for now)
        const waitTime = this.estimateWaitTime(place.rating || 0);

        return {
          id: place.place_id || `hospital_${index}`,
          name: place.name,
          address: place.formatted_address,
          phone: place.formatted_phone_number || "Phone not available",
          hours,
          waitTime,
          specialties,
          coordinates: {
            latitude: place.geometry.location.lat,
            longitude: place.geometry.location.lng,
          },
        };
      });

      return hospitals;
    } catch (error) {
      console.error("Error fetching hospitals from API:", error);
      throw error;
    }
  }

  // Helper method to determine specialties from Google Places types
  private static getSpecialtiesFromTypes(types: string[]): string[] {
    const specialties: string[] = [];
    
    if (types.includes("hospital")) {
      specialties.push("General Hospital");
    }
    if (types.includes("health")) {
      specialties.push("Healthcare");
    }
    if (types.includes("emergency")) {
      specialties.push("Emergency Care");
    }
    if (types.includes("doctor")) {
      specialties.push("Medical Care");
    }
    if (types.includes("dentist")) {
      specialties.push("Dental Care");
    }
    if (types.includes("pharmacy")) {
      specialties.push("Pharmacy");
    }
    
    // Default specialties if none detected
    if (specialties.length === 0) {
      specialties.push("Medical Care", "Emergency Care");
    }
    
    return specialties;
  }

  // Helper method to estimate wait time based on rating
  private static estimateWaitTime(rating: number): string {
    if (rating >= 4.5) return "~10 mins";
    if (rating >= 4.0) return "~20 mins";
    if (rating >= 3.5) return "~30 mins";
    return "~45 mins";
  }

  // Calculate distance between two points using Haversine formula
  static calculateDistance(
    lat1: number,
    lon1: number,
    lat2: number,
    lon2: number
  ): number {
    try {
      // Validate input coordinates
      if (!lat1 || !lon1 || !lat2 || !lon2) {
        return 0;
      }

      const R = 3959; // Earth's radius in miles
      const dLat = (lat2 - lat1) * (Math.PI / 180);
      const dLon = (lon2 - lon1) * (Math.PI / 180);
      const a =
        Math.sin(dLat / 2) * Math.sin(dLat / 2) +
        Math.cos(lat1 * (Math.PI / 180)) *
          Math.cos(lat2 * (Math.PI / 180)) *
          Math.sin(dLon / 2) *
          Math.sin(dLon / 2);
      const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
      return R * c;
    } catch (error) {
      console.error("Error calculating distance:", error);
      return 0;
    }
  }

  // Get nearby hospitals with real API integration
  static async getNearbyHospitals(
    userLocation: LocationType,
    radius: number = 50
  ): Promise<Hospital[]> {
    try {
      // Validate user location
      if (!userLocation || !userLocation.latitude || !userLocation.longitude) {
        throw new Error("Invalid location data. Please try getting your location again.");
      }

      // Validate radius
      if (radius <= 0) {
        radius = 50; // Default to 50 miles
      }

      // Try to fetch from Google Places API first
      try {
        const apiRadius = Math.min(radius * 1609.34, 50000); // Convert miles to meters, max 50km
        const hospitals = await this.fetchHospitalsFromAPI(
          userLocation.latitude,
          userLocation.longitude,
          apiRadius
        );

        // Add distance to each hospital
        const hospitalsWithDistance = hospitals.map((hospital) => {
          const distance = this.calculateDistance(
            userLocation.latitude,
            userLocation.longitude,
            hospital.coordinates.latitude,
            hospital.coordinates.longitude
          );
          return { ...hospital, distance };
        });

        // Filter hospitals within radius and sort by distance
        const nearbyHospitals = hospitalsWithDistance
          .filter((hospital) => hospital.distance! <= radius)
          .sort((a, b) => a.distance! - b.distance!);

        return nearbyHospitals;
      } catch (apiError) {
        console.warn("API failed, using fallback data:", apiError);
        
        // Fallback to mock data if API fails
        const hospitalsWithDistance = FALLBACK_HOSPITALS.map((hospital) => {
          const distance = this.calculateDistance(
            userLocation.latitude,
            userLocation.longitude,
            hospital.coordinates.latitude,
            hospital.coordinates.longitude
          );
          return { ...hospital, distance };
        });

        // Filter hospitals within radius and sort by distance
        const nearbyHospitals = hospitalsWithDistance
          .filter((hospital) => hospital.distance! <= radius)
          .sort((a, b) => a.distance! - b.distance!);

        return nearbyHospitals;
      }
    } catch (error) {
      console.error("Error fetching hospitals:", error);
      
      if (error instanceof Error) {
        throw new Error(error.message);
      } else {
        throw new Error("Unable to fetch nearby hospitals. Please try again.");
      }
    }
  }

  // Search hospitals by name with real API integration
  static async searchHospitals(
    query: string,
    userLocation: LocationType
  ): Promise<Hospital[]> {
    try {
      // Validate search query
      if (!query || !query.trim()) {
        throw new Error("Please enter a search term.");
      }

      // Validate user location
      if (!userLocation || !userLocation.latitude || !userLocation.longitude) {
        throw new Error("Location is required for hospital search. Please get your location first.");
      }

      try {
        // Use Google Places API text search for better results
        const apiRadius = 100000; // 100km radius for search
        const hospitals = await this.fetchHospitalsFromAPI(
          userLocation.latitude,
          userLocation.longitude,
          apiRadius,
          query
        );

        // Add distance to each hospital
        const hospitalsWithDistance = hospitals.map((hospital) => {
          const distance = this.calculateDistance(
            userLocation.latitude,
            userLocation.longitude,
            hospital.coordinates.latitude,
            hospital.coordinates.longitude
          );
          return { ...hospital, distance };
        });

        // Sort by distance
        const searchResults = hospitalsWithDistance.sort((a, b) => a.distance! - b.distance!);

        return searchResults;
      } catch (apiError) {
        console.warn("API search failed, using fallback:", apiError);
        
        // Fallback to local search with nearby hospitals
        const allHospitals = await this.getNearbyHospitals(userLocation, 100);
        const searchTerm = query.trim().toLowerCase();
        
        const searchResults = allHospitals.filter(
          (hospital) =>
            hospital.name.toLowerCase().includes(searchTerm) ||
            hospital.address.toLowerCase().includes(searchTerm) ||
            hospital.specialties.some((specialty) =>
              specialty.toLowerCase().includes(searchTerm)
            )
        );

        return searchResults;
      }
    } catch (error) {
      console.error("Error searching hospitals:", error);
      
      if (error instanceof Error) {
        throw new Error(error.message);
      } else {
        throw new Error("Unable to search hospitals. Please try again.");
      }
    }
  }

  // Get route to hospital (mock implementation)
  static async getRouteToHospital(
    origin: LocationType,
    destination: Hospital
  ): Promise<Route> {
    try {
      // Validate inputs
      if (!origin || !origin.latitude || !origin.longitude) {
        throw new Error("Invalid origin location.");
      }

      if (!destination || !destination.coordinates) {
        throw new Error("Invalid destination hospital.");
      }

      const distance = this.calculateDistance(
        origin.latitude,
        origin.longitude,
        destination.coordinates.latitude,
        destination.coordinates.longitude
      );

      // Mock route data
      return {
        distance,
        duration: Math.round(distance * 2), // Rough estimate: 2 minutes per mile
        polyline: "", // Would contain encoded polyline from Google Maps API
      };
    } catch (error) {
      console.error("Error getting route:", error);
      throw new Error("Unable to calculate route. Please try again.");
    }
  }

  // Open directions in external maps app
  static openDirections(hospital: Hospital): void {
    try {
      if (!hospital || !hospital.coordinates) {
        throw new Error("Invalid hospital data.");
      }

      const { latitude, longitude } = hospital.coordinates;
      const url = `https://www.google.com/maps/dir/?api=1&destination=${latitude},${longitude}`;

      // In a real app, you'd use Linking.openURL(url)
      console.log("Opening directions:", url);
    } catch (error) {
      console.error("Error opening directions:", error);
      throw new Error("Unable to open directions.");
    }
  }

  // Call hospital
  static callHospital(phone: string): void {
    try {
      if (!phone || !phone.trim()) {
        throw new Error("Invalid phone number.");
      }

      const url = `tel:${phone.trim()}`;
      // In a real app, you'd use Linking.openURL(url)
      console.log("Calling hospital:", url);
    } catch (error) {
      console.error("Error calling hospital:", error);
      throw new Error("Unable to make phone call.");
    }
  }
}
