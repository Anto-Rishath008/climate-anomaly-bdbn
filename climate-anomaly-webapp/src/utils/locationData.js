// Location data and search utilities for Aura Weather
export const locationDatabase = [
  // Major Cities - North America
  { name: 'New York, NY, USA', country: 'USA', coords: { lat: 40.7128, lng: -74.0060 }, population: 8336817 },
  { name: 'Los Angeles, CA, USA', country: 'USA', coords: { lat: 34.0522, lng: -118.2437 }, population: 3979576 },
  { name: 'Chicago, IL, USA', country: 'USA', coords: { lat: 41.8781, lng: -87.6298 }, population: 2693976 },
  { name: 'Houston, TX, USA', country: 'USA', coords: { lat: 29.7604, lng: -95.3698 }, population: 2320268 },
  { name: 'Phoenix, AZ, USA', country: 'USA', coords: { lat: 33.4484, lng: -112.0740 }, population: 1680992 },
  { name: 'Philadelphia, PA, USA', country: 'USA', coords: { lat: 39.9526, lng: -75.1652 }, population: 1584064 },
  { name: 'San Antonio, TX, USA', country: 'USA', coords: { lat: 29.4241, lng: -98.4936 }, population: 1547253 },
  { name: 'San Diego, CA, USA', country: 'USA', coords: { lat: 32.7157, lng: -117.1611 }, population: 1423851 },
  { name: 'Dallas, TX, USA', country: 'USA', coords: { lat: 32.7767, lng: -96.7970 }, population: 1343573 },
  { name: 'San Jose, CA, USA', country: 'USA', coords: { lat: 37.3382, lng: -121.8863 }, population: 1021795 },
  { name: 'Austin, TX, USA', country: 'USA', coords: { lat: 30.2672, lng: -97.7431 }, population: 978908 },
  { name: 'Jacksonville, FL, USA', country: 'USA', coords: { lat: 30.3322, lng: -81.6557 }, population: 911507 },
  { name: 'Fort Worth, TX, USA', country: 'USA', coords: { lat: 32.7555, lng: -97.3308 }, population: 909585 },
  { name: 'Columbus, OH, USA', country: 'USA', coords: { lat: 39.9612, lng: -82.9988 }, population: 898553 },
  { name: 'San Francisco, CA, USA', country: 'USA', coords: { lat: 37.7749, lng: -122.4194 }, population: 881549 },
  { name: 'Charlotte, NC, USA', country: 'USA', coords: { lat: 35.2271, lng: -80.8431 }, population: 885708 },
  { name: 'Indianapolis, IN, USA', country: 'USA', coords: { lat: 39.7684, lng: -86.1581 }, population: 876384 },
  { name: 'Seattle, WA, USA', country: 'USA', coords: { lat: 47.6062, lng: -122.3321 }, population: 753675 },
  { name: 'Denver, CO, USA', country: 'USA', coords: { lat: 39.7392, lng: -104.9903 }, population: 715522 },
  { name: 'Boston, MA, USA', country: 'USA', coords: { lat: 42.3601, lng: -71.0589 }, population: 692600 },
  { name: 'Toronto, ON, Canada', country: 'Canada', coords: { lat: 43.6532, lng: -79.3832 }, population: 2930000 },
  { name: 'Montreal, QC, Canada', country: 'Canada', coords: { lat: 45.5017, lng: -73.5673 }, population: 1780000 },
  { name: 'Vancouver, BC, Canada', country: 'Canada', coords: { lat: 49.2827, lng: -123.1207 }, population: 675218 },
  { name: 'Calgary, AB, Canada', country: 'Canada', coords: { lat: 51.0447, lng: -114.0719 }, population: 1336000 },
  { name: 'Ottawa, ON, Canada', country: 'Canada', coords: { lat: 45.4215, lng: -75.6972 }, population: 994837 },
  { name: 'Mexico City, Mexico', country: 'Mexico', coords: { lat: 19.4326, lng: -99.1332 }, population: 9209944 },

  // Europe
  { name: 'London, England, UK', country: 'UK', coords: { lat: 51.5074, lng: -0.1278 }, population: 9304016 },
  { name: 'Paris, France', country: 'France', coords: { lat: 48.8566, lng: 2.3522 }, population: 2165423 },
  { name: 'Berlin, Germany', country: 'Germany', coords: { lat: 52.5200, lng: 13.4050 }, population: 3669491 },
  { name: 'Madrid, Spain', country: 'Spain', coords: { lat: 40.4168, lng: -3.7038 }, population: 3223334 },
  { name: 'Rome, Italy', country: 'Italy', coords: { lat: 41.9028, lng: 12.4964 }, population: 2873494 },
  { name: 'Amsterdam, Netherlands', country: 'Netherlands', coords: { lat: 52.3676, lng: 4.9041 }, population: 872680 },
  { name: 'Vienna, Austria', country: 'Austria', coords: { lat: 48.2082, lng: 16.3738 }, population: 1911191 },
  { name: 'Stockholm, Sweden', country: 'Sweden', coords: { lat: 59.3293, lng: 18.0686 }, population: 975551 },
  { name: 'Copenhagen, Denmark', country: 'Denmark', coords: { lat: 55.6761, lng: 12.5683 }, population: 648731 },
  { name: 'Oslo, Norway', country: 'Norway', coords: { lat: 59.9139, lng: 10.7522 }, population: 697549 },
  { name: 'Helsinki, Finland', country: 'Finland', coords: { lat: 60.1699, lng: 24.9384 }, population: 658864 },
  { name: 'Prague, Czech Republic', country: 'Czech Republic', coords: { lat: 50.0755, lng: 14.4378 }, population: 1335084 },
  { name: 'Warsaw, Poland', country: 'Poland', coords: { lat: 52.2297, lng: 21.0122 }, population: 1793579 },
  { name: 'Budapest, Hungary', country: 'Hungary', coords: { lat: 47.4979, lng: 19.0402 }, population: 1752286 },
  { name: 'Zurich, Switzerland', country: 'Switzerland', coords: { lat: 47.3769, lng: 8.5417 }, population: 415367 },
  { name: 'Brussels, Belgium', country: 'Belgium', coords: { lat: 50.8503, lng: 4.3517 }, population: 1208542 },

  // Asia
  { name: 'Tokyo, Japan', country: 'Japan', coords: { lat: 35.6762, lng: 139.6503 }, population: 37400068 },
  { name: 'Shanghai, China', country: 'China', coords: { lat: 31.2304, lng: 121.4737 }, population: 27058479 },
  { name: 'Beijing, China', country: 'China', coords: { lat: 39.9042, lng: 116.4074 }, population: 21542000 },
  { name: 'Mumbai, India', country: 'India', coords: { lat: 19.0760, lng: 72.8777 }, population: 20411274 },
  { name: 'Delhi, India', country: 'India', coords: { lat: 28.7041, lng: 77.1025 }, population: 31400000 },
  { name: 'Seoul, South Korea', country: 'South Korea', coords: { lat: 37.5665, lng: 126.9780 }, population: 9720846 },
  { name: 'Singapore', country: 'Singapore', coords: { lat: 1.3521, lng: 103.8198 }, population: 5850342 },
  { name: 'Hong Kong', country: 'Hong Kong', coords: { lat: 22.3193, lng: 114.1694 }, population: 7496981 },
  { name: 'Bangkok, Thailand', country: 'Thailand', coords: { lat: 13.7563, lng: 100.5018 }, population: 10156000 },
  { name: 'Jakarta, Indonesia', country: 'Indonesia', coords: { lat: -6.2088, lng: 106.8456 }, population: 10770487 },
  { name: 'Manila, Philippines', country: 'Philippines', coords: { lat: 14.5995, lng: 120.9842 }, population: 13482462 },
  { name: 'Kuala Lumpur, Malaysia', country: 'Malaysia', coords: { lat: 3.1390, lng: 101.6869 }, population: 1808000 },
  { name: 'Bangalore, India', country: 'India', coords: { lat: 12.9716, lng: 77.5946 }, population: 12326532 },
  { name: 'Chennai, India', country: 'India', coords: { lat: 13.0827, lng: 80.2707 }, population: 7088000 },
  { name: 'Kolkata, India', country: 'India', coords: { lat: 22.5726, lng: 88.3639 }, population: 4496694 },
  { name: 'Hyderabad, India', country: 'India', coords: { lat: 17.3850, lng: 78.4867 }, population: 6809970 },

  // Australia & Oceania
  { name: 'Sydney, NSW, Australia', country: 'Australia', coords: { lat: -33.8688, lng: 151.2093 }, population: 5312163 },
  { name: 'Melbourne, VIC, Australia', country: 'Australia', coords: { lat: -37.8136, lng: 144.9631 }, population: 5078193 },
  { name: 'Brisbane, QLD, Australia', country: 'Australia', coords: { lat: -27.4698, lng: 153.0251 }, population: 2560720 },
  { name: 'Perth, WA, Australia', country: 'Australia', coords: { lat: -31.9505, lng: 115.8605 }, population: 2125114 },
  { name: 'Adelaide, SA, Australia', country: 'Australia', coords: { lat: -34.9285, lng: 138.6007 }, population: 1359760 },
  { name: 'Auckland, New Zealand', country: 'New Zealand', coords: { lat: -36.8485, lng: 174.7633 }, population: 1657200 },
  { name: 'Wellington, New Zealand', country: 'New Zealand', coords: { lat: -41.2865, lng: 174.7762 }, population: 418500 },

  // South America
  { name: 'São Paulo, Brazil', country: 'Brazil', coords: { lat: -23.5505, lng: -46.6333 }, population: 12325232 },
  { name: 'Rio de Janeiro, Brazil', country: 'Brazil', coords: { lat: -22.9068, lng: -43.1729 }, population: 6747815 },
  { name: 'Buenos Aires, Argentina', country: 'Argentina', coords: { lat: -34.6118, lng: -58.3960 }, population: 3054300 },
  { name: 'Lima, Peru', country: 'Peru', coords: { lat: -12.0464, lng: -77.0428 }, population: 10092000 },
  { name: 'Bogotá, Colombia', country: 'Colombia', coords: { lat: 4.7110, lng: -74.0721 }, population: 7412566 },
  { name: 'Santiago, Chile', country: 'Chile', coords: { lat: -33.4489, lng: -70.6693 }, population: 6680000 },
  { name: 'Caracas, Venezuela', country: 'Venezuela', coords: { lat: 10.4806, lng: -66.9036 }, population: 2935744 },

  // Africa
  { name: 'Cairo, Egypt', country: 'Egypt', coords: { lat: 30.0444, lng: 31.2357 }, population: 20484965 },
  { name: 'Lagos, Nigeria', country: 'Nigeria', coords: { lat: 6.5244, lng: 3.3792 }, population: 14862000 },
  { name: 'Casablanca, Morocco', country: 'Morocco', coords: { lat: 33.5731, lng: -7.5898 }, population: 3359818 },
  { name: 'Cape Town, South Africa', country: 'South Africa', coords: { lat: -33.9249, lng: 18.4241 }, population: 4618000 },
  { name: 'Johannesburg, South Africa', country: 'South Africa', coords: { lat: -26.2041, lng: 28.0473 }, population: 4949347 },
  { name: 'Nairobi, Kenya', country: 'Kenya', coords: { lat: -1.2921, lng: 36.8219 }, population: 4397073 },
  { name: 'Addis Ababa, Ethiopia', country: 'Ethiopia', coords: { lat: 9.1450, lng: 38.7451 }, population: 3040740 },

  // Middle East
  { name: 'Dubai, UAE', country: 'UAE', coords: { lat: 25.2048, lng: 55.2708 }, population: 3411200 },
  { name: 'Abu Dhabi, UAE', country: 'UAE', coords: { lat: 24.4539, lng: 54.3773 }, population: 1482816 },
  { name: 'Riyadh, Saudi Arabia', country: 'Saudi Arabia', coords: { lat: 24.7136, lng: 46.6753 }, population: 7676654 },
  { name: 'Tel Aviv, Israel', country: 'Israel', coords: { lat: 32.0853, lng: 34.7818 }, population: 460613 },
  { name: 'Jerusalem, Israel', country: 'Israel', coords: { lat: 31.7683, lng: 35.2137 }, population: 936425 },
  { name: 'Istanbul, Turkey', country: 'Turkey', coords: { lat: 41.0082, lng: 28.9784 }, population: 15519267 },
  { name: 'Ankara, Turkey', country: 'Turkey', coords: { lat: 39.9334, lng: 32.8597 }, population: 5663322 },
  { name: 'Tehran, Iran', country: 'Iran', coords: { lat: 35.6892, lng: 51.3890 }, population: 9135000 },

  // Additional popular cities
  { name: 'Las Vegas, NV, USA', country: 'USA', coords: { lat: 36.1699, lng: -115.1398 }, population: 651319 },
  { name: 'Miami, FL, USA', country: 'USA', coords: { lat: 25.7617, lng: -80.1918 }, population: 467963 },
  { name: 'Nashville, TN, USA', country: 'USA', coords: { lat: 36.1627, lng: -86.7816 }, population: 694144 },
  { name: 'Atlanta, GA, USA', country: 'USA', coords: { lat: 33.7490, lng: -84.3880 }, population: 506811 },
  { name: 'Portland, OR, USA', country: 'USA', coords: { lat: 45.5152, lng: -122.6784 }, population: 652503 },
  { name: 'Minneapolis, MN, USA', country: 'USA', coords: { lat: 44.9778, lng: -93.2650 }, population: 429954 },
  { name: 'New Orleans, LA, USA', country: 'USA', coords: { lat: 29.9511, lng: -90.0715 }, population: 383997 },
  { name: 'Salt Lake City, UT, USA', country: 'USA', coords: { lat: 40.7608, lng: -111.8910 }, population: 200567 }
];

// Search function with fuzzy matching
export const searchLocations = (query, limit = 8) => {
  if (!query || query.length < 1) return [];
  
  const normalizedQuery = query.toLowerCase().trim();
  
  // Exact matches first
  const exactMatches = locationDatabase.filter(location =>
    location.name.toLowerCase().startsWith(normalizedQuery)
  );
  
  // Partial matches
  const partialMatches = locationDatabase.filter(location =>
    !location.name.toLowerCase().startsWith(normalizedQuery) &&
    location.name.toLowerCase().includes(normalizedQuery)
  );
  
  // Country matches
  const countryMatches = locationDatabase.filter(location =>
    !location.name.toLowerCase().includes(normalizedQuery) &&
    location.country.toLowerCase().includes(normalizedQuery)
  );
  
  // Combine and deduplicate
  const results = [...exactMatches, ...partialMatches, ...countryMatches];
  const uniqueResults = results.filter((location, index, self) =>
    index === self.findIndex(l => l.name === location.name)
  );
  
  // Sort by population (bigger cities first) and limit results
  return uniqueResults
    .sort((a, b) => b.population - a.population)
    .slice(0, limit);
};

// Get popular cities for initial suggestions
export const getPopularCities = (limit = 10) => {
  return locationDatabase
    .sort((a, b) => b.population - a.population)
    .slice(0, limit);
};

// Get random city suggestions
export const getRandomCities = (limit = 6) => {
  const shuffled = [...locationDatabase].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, limit);
};