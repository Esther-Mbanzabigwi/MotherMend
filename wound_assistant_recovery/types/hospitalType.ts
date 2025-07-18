export interface Hospital {
  id: string;
  name: string;
  address: string;
  phone: string;
  hours: string;
  waitTime: string;
  specialties: string[];
  coordinates: {
    latitude: number;
    longitude: number;
  };
  distance?: number;
}

export interface Location {
  latitude: number;
  longitude: number;
  address?: string;
}

export interface Route {
  distance: number;
  duration: number;
  polyline: string;
}
