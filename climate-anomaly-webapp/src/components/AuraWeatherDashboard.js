import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FaThermometerHalf, 
  FaCloudRain, 
  FaWind, 
  FaSearch,
  FaMapMarkerAlt,
  FaClock,
  FaGlobe,
  FaLocationArrow
} from 'react-icons/fa';
import { theme, threatLevels, weatherIcons } from '../theme';
import { searchLocations, getPopularCities, getRandomCities } from '../utils/locationData';
import InteractiveWorldMap from './InteractiveWorldMap';

const DashboardContainer = styled.div`
  padding: 2rem;
  max-width: 1600px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 2rem;
  min-height: calc(100vh - 160px);
  
  /* Debug: Ensure grid items are visible */
  & > * {
    min-height: 200px;
  }
  
  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }
`;

const MapSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
`;

const SearchBar = styled.div`
  position: relative;
  
  input {
    width: 100%;
    padding: 1rem 1rem 1rem 3rem;
    background: ${theme.glassmorphism.light.background};
    backdrop-filter: ${theme.glassmorphism.light.backdropFilter};
    border: ${theme.glassmorphism.light.border};
    border-radius: ${theme.borderRadius.lg};
    color: ${theme.colors.text.primary};
    font-size: 1.1rem;
    transition: all ${theme.animation.duration.normal} ease;
    
    &::placeholder {
      color: ${theme.colors.text.muted};
    }
    
    &:focus {
      outline: none;
      border-color: ${theme.colors.accent.info};
      background: ${theme.glassmorphism.medium.background};
      box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.1);
    }
  }
  
  .search-icon {
    position: absolute;
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
    color: ${theme.colors.text.muted};
    font-size: 1.2rem;
  }
`;

const SearchSuggestions = styled(motion.div)`
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: ${theme.glassmorphism.medium.background};
  backdrop-filter: ${theme.glassmorphism.medium.backdropFilter};
  border: ${theme.glassmorphism.medium.border};
  border-radius: ${theme.borderRadius.lg};
  margin-top: 0.5rem;
  max-height: 400px;
  overflow-y: auto;
  z-index: 1000;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);

  .suggestion-section {
    padding: 1rem 0;
    
    &:not(:last-child) {
      border-bottom: 1px solid ${theme.colors.border.primary};
    }
    
    .section-title {
      padding: 0.5rem 1rem;
      font-size: 0.9rem;
      font-weight: ${theme.typography.fontWeight.semibold};
      color: ${theme.colors.text.secondary};
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
  }
  
  .suggestion-item {
    display: flex;
    align-items: center;
    padding: 0.75rem 1rem;
    cursor: pointer;
    transition: all ${theme.animation.duration.fast} ease;
    
    &:hover {
      background: ${theme.glassmorphism.light.background};
    }
    
    .location-icon {
      color: ${theme.colors.accent.info};
      margin-right: 0.75rem;
      font-size: 1rem;
    }
    
    .location-info {
      flex: 1;
      
      .location-name {
        color: ${theme.colors.text.primary};
        font-weight: ${theme.typography.fontWeight.medium};
        margin-bottom: 0.2rem;
      }
      
      .location-details {
        color: ${theme.colors.text.muted};
        font-size: 0.85rem;
      }
    }
    
    .population {
      color: ${theme.colors.text.muted};
      font-size: 0.8rem;
      opacity: 0.7;
    }
  }
`;

const MapContainer = styled(motion.div)`
  background: ${theme.glassmorphism.light.background};
  backdrop-filter: ${theme.glassmorphism.light.backdropFilter};
  border: ${theme.glassmorphism.light.border};
  border-radius: ${theme.borderRadius.xl};
  padding: 1.5rem;
  flex: 1;
  min-height: 500px;
  position: relative;
  overflow: hidden;
  
  .map-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: ${theme.colors.text.secondary};
    font-size: 1.2rem;
    text-align: center;
    
    .globe-icon {
      font-size: 4rem;
      margin-bottom: 1rem;
      color: ${theme.colors.accent.info};
    }
  }
`;

const RightPanel = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  min-width: 350px;
  max-width: 400px;
  
  /* Debug: Temporary border to see panel */
  border: 2px solid rgba(96, 165, 250, 0.3);
  border-radius: 16px;
  padding: 1rem;
  
  @media (max-width: 1024px) {
    min-width: unset;
    max-width: unset;
  }
`;

const WeatherCard = styled(motion.div)`
  background: ${theme?.glassmorphism?.medium?.background || 'rgba(255, 255, 255, 0.15)'};
  backdrop-filter: ${theme?.glassmorphism?.medium?.backdropFilter || 'blur(20px)'};
  border: ${theme?.glassmorphism?.medium?.border || '1px solid rgba(255, 255, 255, 0.2)'};
  border-radius: ${theme?.borderRadius?.xl || '16px'};
  padding: 2rem;
  min-height: 300px;
  
  .location {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: ${theme.colors.text.secondary};
    margin-bottom: 1.5rem;
    
    .location-icon {
      color: ${theme.colors.accent.info};
    }
  }
  
  .current-weather {
    text-align: center;
    margin-bottom: 2rem;
    
    .weather-icon {
      font-size: 4rem;
      margin-bottom: 1rem;
    }
    
    .temperature {
      font-size: 3rem;
      font-weight: ${theme.typography.fontWeight.light};
      color: ${theme.colors.text.primary};
      margin-bottom: 0.5rem;
    }
    
    .condition {
      color: ${theme.colors.text.secondary};
      font-size: 1.1rem;
    }
  }
  
  .weather-details {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    
    .detail-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.25rem;
      color: ${theme?.colors?.text?.secondary || 'rgba(255, 255, 255, 0.8)'};
      text-align: center;
      
      .detail-icon {
        color: ${theme?.colors?.accent?.info || '#60a5fa'};
        font-size: 1.2rem;
      }
      
      .detail-label {
        font-size: 0.85rem;
      }
      
      .detail-value {
        color: ${theme?.colors?.text?.primary || '#ffffff'};
        font-weight: ${theme?.typography?.fontWeight?.medium || '500'};
        font-size: 0.95rem;
      }
    }
  }
`;

const PredictionPanel = styled(motion.div)`
  background: ${theme.glassmorphism.light.background};
  backdrop-filter: ${theme.glassmorphism.light.backdropFilter};
  border: ${theme.glassmorphism.light.border};
  border-radius: ${theme.borderRadius.xl};
  padding: 2rem;
  
  h3 {
    color: ${theme.colors.text.primary};
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    
    .prediction-icon {
      color: ${theme.colors.accent.info};
    }
  }
  
  .prediction-form {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 1.5rem;
    
    input {
      padding: 0.8rem 1rem;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: ${theme.borderRadius.md};
      color: ${theme.colors.text.primary};
      
      &::placeholder {
        color: ${theme.colors.text.muted};
      }
      
      &:focus {
        outline: none;
        border-color: ${theme.colors.accent.info};
        background: rgba(255, 255, 255, 0.15);
      }
    }
    
    button {
      padding: 0.8rem 1.5rem;
      background: linear-gradient(135deg, ${theme.colors.accent.info} 0%, ${theme.colors.primary[600]} 100%);
      border: none;
      border-radius: ${theme.borderRadius.md};
      color: white;
      font-weight: ${theme.typography.fontWeight.medium};
      cursor: pointer;
      transition: all ${theme.animation.duration.normal} ease;
      
      &:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(96, 165, 250, 0.3);
      }
      
      &:disabled {
        opacity: 0.7;
        cursor: not-allowed;
        transform: none;
      }
    }
  }
`;

const AnomalyReport = styled(motion.div)`
  background: ${theme.glassmorphism.medium.background};
  backdrop-filter: ${theme.glassmorphism.medium.backdropFilter};
  border: ${theme.glassmorphism.medium.border};
  border-radius: ${theme.borderRadius.xl};
  padding: 2rem;
  
  .report-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    
    h3 {
      color: ${theme.colors.text.primary};
      margin: 0;
    }
    
    .threat-badge {
      padding: 0.5rem 1rem;
      border-radius: 20px;
      font-size: 0.9rem;
      font-weight: ${theme.typography.fontWeight.medium};
    }
  }
  
  .report-content {
    color: ${theme.colors.text.secondary};
    line-height: 1.6;
    
    .prediction-summary {
      font-size: 1.1rem;
      margin-bottom: 1rem;
      color: ${theme.colors.text.primary};
    }
    
    .details {
      font-size: 0.95rem;
      white-space: pre-line;
    }
  }
  
  .report-actions {
    display: flex;
    gap: 1rem;
    margin-top: 1.5rem;
    
    button {
      padding: 0.8rem 1.5rem;
      border: none;
      border-radius: ${theme.borderRadius.md};
      font-weight: ${theme.typography.fontWeight.medium};
      cursor: pointer;
      transition: all ${theme.animation.duration.normal} ease;
      
      &.details-btn {
        background: linear-gradient(135deg, ${theme.colors.accent.info} 0%, ${theme.colors.primary[600]} 100%);
        color: white;
        
        &:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 16px rgba(96, 165, 250, 0.3);
        }
      }
      
      &.track-btn {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: ${theme.colors.text.primary};
        
        &:hover {
          background: rgba(255, 255, 255, 0.2);
          transform: translateY(-2px);
        }
      }
    }
  }
`;

function Dashboard() {
  const [selectedLocation, setSelectedLocation] = useState({
    name: 'New York, NY, USA',
    coords: { lat: 40.7128, lng: -74.0060 }
  });
  
  const [currentWeather, setCurrentWeather] = useState({
    temperature: 24,
    feelsLike: 27,
    condition: 'Partly Cloudy',
    icon: weatherIcons?.partlyCloudy || '⛅',
    humidity: 47,
    windSpeed: 19,
    windDirection: 'NW',
    precipitation: 20
  });

  const [predictionQuery, setPredictionQuery] = useState({
    date: '',
    time: ''
  });

  const [predictionResult, setPredictionResult] = useState({
    isVisible: false,
    threatLevel: 'safe',
    summary: '',
    details: ''
  });

  const [searchQuery, setSearchQuery] = useState('');
  const [searchSuggestions, setSearchSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [popularCities] = useState(getPopularCities(8));
  const [randomCities] = useState(getRandomCities(6));
  const [isLoading, setIsLoading] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const [isTracking, setIsTracking] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking'); // 'online', 'offline', 'checking'

  // Handle search input changes
  const handleSearchChange = (e) => {
    const query = e.target.value;
    setSearchQuery(query);
    
    if (query.length > 0) {
      const suggestions = searchLocations(query, 8);
      setSearchSuggestions(suggestions);
      setShowSuggestions(true);
    } else {
      setSearchSuggestions([]);
      setShowSuggestions(false);
    }
  };

  // Handle search submit (Enter key or search button)
  const handleSearchSubmit = async (e) => {
    e.preventDefault();
    if (searchQuery.trim().length === 0) return;

    // Check if it's coordinates (lat, lng format)
    const coordsMatch = searchQuery.match(/^(-?\d+\.?\d*),?\s*(-?\d+\.?\d*)$/);
    if (coordsMatch) {
      const lat = parseFloat(coordsMatch[1]);
      const lng = parseFloat(coordsMatch[2]);
      
      if (lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180) {
        const newLocation = {
          name: `${lat.toFixed(4)}, ${lng.toFixed(4)}`,
          coords: { lat, lng }
        };
        handleLocationSelect(newLocation);
        return;
      }
    }

    // Otherwise, try to geocode the location name
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery)}&limit=1&addressdetails=1`
      );
      
      if (response.ok) {
        const data = await response.json();
        if (data.length > 0) {
          const result = data[0];
          const newLocation = {
            name: result.display_name,
            coords: { lat: parseFloat(result.lat), lng: parseFloat(result.lon) }
          };
          handleLocationSelect(newLocation);
        }
      }
    } catch (error) {
      console.error('Geocoding error:', error);
    }
  };

  // Handle location selection from search or map
  const handleLocationSelect = (location) => {
    setSelectedLocation({
      name: location.name,
      coords: location.coords
    });
    
    // Update search query with location name or coordinates
    const displayName = location.name || `${location.coords.lat.toFixed(4)}, ${location.coords.lng.toFixed(4)}`;
    setSearchQuery(displayName);
    setShowSuggestions(false);
    
    // Simulate weather data update
    setCurrentWeather(prev => ({
      ...prev,
      temperature: Math.round(Math.random() * 30 + 5),
      feelsLike: Math.round(Math.random() * 35 + 5),
      humidity: Math.round(Math.random() * 60 + 30),
      windSpeed: Math.round(Math.random() * 25 + 5),
      precipitation: Math.round(Math.random() * 80)
    }));
  };

  // Handle search focus
  const handleSearchFocus = () => {
    if (searchQuery.length === 0) {
      setShowSuggestions(true);
    }
  };

  // Handle search blur with delay to allow clicking suggestions
  const handleSearchBlur = () => {
    setTimeout(() => {
      setShowSuggestions(false);
    }, 200);
  };



  // Format population for display
  const formatPopulation = (population) => {
    if (population >= 1000000) {
      return `${(population / 1000000).toFixed(1)}M`;
    } else if (population >= 1000) {
      return `${Math.round(population / 1000)}K`;
    }
    return population.toString();
  };

  // Check API status
  const checkApiStatus = React.useCallback(async () => {
    try {
      const response = await fetch('http://localhost:5000/api/models/info', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        setApiStatus('online');
        console.log('✅ API Server is online');
      } else {
        setApiStatus('offline');
      }
    } catch (error) {
      setApiStatus('offline');
      console.log('❌ API Server is offline');
    }
  }, []);



  useEffect(() => {
    // Initialize with current date/time
    const now = new Date();
    const today = now.toISOString().split('T')[0];
    const currentTime = now.toTimeString().slice(0, 5);
    
    setPredictionQuery({
      date: today,
      time: currentTime
    });

    // Check API status on mount
    checkApiStatus();
    
    // Check API status periodically
    const statusInterval = setInterval(checkApiStatus, 30000); // Check every 30 seconds
    
    return () => clearInterval(statusInterval);
  }, [checkApiStatus]);



  const handlePredictionSubmit = async () => {
    if (!predictionQuery.date || !predictionQuery.time) return;
    
    setIsLoading(true);
    
    try {
      // 🧠 STEP 1: TIME-BASED FEATURE ENGINEERING
      // Convert date/time to numerical features for ML model
      const targetDate = new Date(`${predictionQuery.date}T${predictionQuery.time}`);
      const currentDate = new Date();
      
      // Calculate time-based features
      const timeDelta = (targetDate - currentDate) / (1000 * 60 * 60); // hours from now
      const dayOfYear = Math.floor((targetDate - new Date(targetDate.getFullYear(), 0, 0)) / (1000 * 60 * 60 * 24));
      const hourOfDay = targetDate.getHours();
      const seasonalFactor = Math.sin(2 * Math.PI * dayOfYear / 365); // Seasonal variation
      const diurnalFactor = Math.sin(2 * Math.PI * hourOfDay / 24); // Daily cycle
      
      // 🌡️ STEP 2: PREDICT FUTURE WEATHER CONDITIONS
      // Use time-based patterns to predict future weather
      const baseTemp = currentWeather.temperature;
      const predictedTemp = baseTemp + 
        (seasonalFactor * 5) + // Seasonal temperature change
        (diurnalFactor * 8) + // Daily temperature cycle
        (Math.random() - 0.5) * 4; // Natural variation
      
      const predictedHumidity = Math.max(20, Math.min(90, 
        currentWeather.humidity + 
        (seasonalFactor * 15) + 
        (diurnalFactor * -10) + // Humidity typically lower during day
        (Math.random() - 0.5) * 20
      ));
      
      const predictedWindSpeed = Math.max(0, 
        currentWeather.windSpeed + 
        (Math.abs(diurnalFactor) * 5) + // Wind often stronger during day
        (Math.random() - 0.5) * 10
      );
      
      const predictedPrecipitation = Math.max(0, Math.min(100,
        currentWeather.precipitation + 
        (seasonalFactor > 0 ? 20 : -10) + // Higher rain in warmer seasons
        (Math.random() - 0.3) * 30
      ));
      
      // 🧠 STEP 3: CALL CLIMATE ANOMALY DETECTION MODEL
      console.log('🔄 Sending prediction request to model...', {
        temperature: predictedTemp.toFixed(2),
        humidity: predictedHumidity.toFixed(1),
        location: selectedLocation?.name || 'Unknown'
      });
      
      const response = await fetch('http://localhost:5000/api/predict/climate-anomaly', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          temperature: predictedTemp,
          humidity: predictedHumidity,
          pressure: 1013.25 + (Math.random() - 0.5) * 20, // Pressure variation
          wind_speed: predictedWindSpeed,
          precipitation: predictedPrecipitation / 100, // Convert to fraction
          time_features: {
            hours_from_now: timeDelta,
            day_of_year: dayOfYear,
            hour_of_day: hourOfDay,
            seasonal_factor: seasonalFactor,
            diurnal_factor: diurnalFactor
          }
        }),
      });
      
      if (!response.ok) {
        throw new Error(`API Error: ${response.status} - ${response.statusText}`);
      }

      const result = await response.json();
      console.log('✅ Model prediction response:', result);
      
      // Validate model response
      if (!result || typeof result.anomaly_score === 'undefined') {
        throw new Error('Invalid model response: missing anomaly_score');
      }
      
      // 📊 STEP 4: INTERPRET MODEL RESULTS
      let threatLevel = 'safe';
      let summary = 'Normal Weather Conditions';
      let details = `Weather prediction for ${targetDate.toLocaleDateString()} at ${targetDate.toLocaleTimeString()}:\n\n`;
      
      // Add predicted weather details
      details += `🌡️ Temperature: ${predictedTemp.toFixed(1)}°C\n`;
      details += `💧 Humidity: ${predictedHumidity.toFixed(0)}%\n`;
      details += `💨 Wind Speed: ${predictedWindSpeed.toFixed(1)} km/h\n`;
      details += `🌧️ Rain Chance: ${predictedPrecipitation.toFixed(0)}%\n\n`;
      
      // Analyze anomaly detection results
      if (result.is_anomaly && result.anomaly_score > 0.8) {
        threatLevel = 'high';
        summary = 'High Probability of Extreme Weather';
        details += `🚨 ANOMALY DETECTED: ${(result.anomaly_score * 100).toFixed(1)}% anomaly score\n`;
        details += `⚠️ Confidence: ${(result.confidence * 100).toFixed(1)}%\n\n`;
        details += `Extreme weather conditions predicted. Prepare for potential severe weather events.`;
      } else if (result.is_anomaly && result.anomaly_score > 0.5) {
        threatLevel = 'moderate';
        summary = 'Unusual Weather Patterns Expected';
        details += `⚠️ Moderate anomaly detected: ${(result.anomaly_score * 100).toFixed(1)}% score\n`;
        details += `📊 Confidence: ${(result.confidence * 100).toFixed(1)}%\n\n`;
        details += `Weather conditions may deviate from normal patterns. Monitor closely.`;
      } else {
        details += `✅ Normal conditions expected\n`;
        details += `📊 Anomaly score: ${(result.anomaly_score * 100).toFixed(1)}%\n`;
        details += `🎯 Confidence: ${(result.confidence * 100).toFixed(1)}%`;
      }
      
      // 🔄 STEP 5: UPDATE PREDICTED WEATHER DISPLAY
      setCurrentWeather(prev => ({
        ...prev,
        temperature: Math.round(predictedTemp),
        feelsLike: Math.round(predictedTemp + (Math.random() - 0.5) * 4),
        humidity: Math.round(predictedHumidity),
        windSpeed: Math.round(predictedWindSpeed),
        precipitation: Math.round(predictedPrecipitation)
      }));

      setPredictionResult({
        isVisible: true,
        threatLevel,
        summary,
        details
      });
    } catch (error) {
      console.error('🚨 Prediction error:', error);
      
      let errorMessage = 'There was an error connecting to the prediction service. Please try again later.';
      let errorDetails = `Error: ${error.message}`;
      
      if (error.message.includes('Failed to fetch')) {
        errorMessage = 'Cannot connect to climate model API';
        errorDetails = `🔌 Connection Error: Make sure the Flask API server is running on http://localhost:5000
        
📋 To fix this:
1. Check if Python server is running
2. Verify API endpoint is accessible
3. Check network connectivity
        
⚠️ Using simulated prediction data for demonstration.`;
        
        // Provide simulated data when API is unavailable
        const simulatedResult = {
          anomaly_score: Math.random() * 0.8,
          confidence: 0.7 + Math.random() * 0.2,
          is_anomaly: Math.random() > 0.6
        };
        
        let threatLevel = simulatedResult.is_anomaly ? 'moderate' : 'safe';
        let summary = simulatedResult.is_anomaly ? 'Potential Weather Anomaly (Simulated)' : 'Normal Conditions Expected (Simulated)';
        
        setPredictionResult({
          isVisible: true,
          threatLevel,
          summary,
          details: errorDetails
        });
        
        setIsLoading(false);
        return;
      }
      
      setPredictionResult({
        isVisible: true,
        threatLevel: 'moderate',
        summary: 'Unable to Generate Prediction',
        details: errorDetails
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getThreatLevelStyle = (level) => {
    const threat = threatLevels[level] || threatLevels.safe;
    return {
      backgroundColor: threat.background,
      color: threat.color,
      border: `1px solid ${threat.color}30`
    };
  };

  return (
    <DashboardContainer>
      <MapSection>
        <SearchBar>
          <form onSubmit={handleSearchSubmit} style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
            <FaSearch className="search-icon" />
            <input
              type="text"
              placeholder="Search for any location worldwide or enter coordinates (lat, lng)..."
              value={searchQuery}
              onChange={handleSearchChange}
              onFocus={handleSearchFocus}
              onBlur={handleSearchBlur}
            />
          </form>
          
          <AnimatePresence>
            {showSuggestions && (
              <SearchSuggestions
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.2 }}
              >
                {searchQuery.length > 0 ? (
                  // Search results
                  searchSuggestions.length > 0 ? (
                    <div className="suggestion-section">
                      <div className="section-title">Search Results</div>
                      {searchSuggestions.map((location, index) => (
                        <div
                          key={`${location.name}-${index}`}
                          className="suggestion-item"
                          onClick={() => handleLocationSelect(location)}
                        >
                          <FaLocationArrow className="location-icon" />
                          <div className="location-info">
                            <div className="location-name">{location.name}</div>
                            <div className="location-details">{location.country}</div>
                          </div>
                          <div className="population">{formatPopulation(location.population)}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="suggestion-section">
                      <div className="section-title">No Results Found</div>
                      <div style={{ padding: '1rem', color: theme.colors.text.muted }}>
                        Try searching for a city, country, or region name
                      </div>
                    </div>
                  )
                ) : (
                  // Popular and random cities when no search query
                  <>
                    <div className="suggestion-section">
                      <div className="section-title">Popular Cities</div>
                      {popularCities.slice(0, 5).map((location, index) => (
                        <div
                          key={`popular-${location.name}-${index}`}
                          className="suggestion-item"
                          onClick={() => handleLocationSelect(location)}
                        >
                          <FaMapMarkerAlt className="location-icon" />
                          <div className="location-info">
                            <div className="location-name">{location.name}</div>
                            <div className="location-details">{location.country}</div>
                          </div>
                          <div className="population">{formatPopulation(location.population)}</div>
                        </div>
                      ))}
                    </div>
                    
                    <div className="suggestion-section">
                      <div className="section-title">Explore</div>
                      {randomCities.map((location, index) => (
                        <div
                          key={`random-${location.name}-${index}`}
                          className="suggestion-item"
                          onClick={() => handleLocationSelect(location)}
                        >
                          <FaGlobe className="location-icon" />
                          <div className="location-info">
                            <div className="location-name">{location.name}</div>
                            <div className="location-details">{location.country}</div>
                          </div>
                          <div className="population">{formatPopulation(location.population)}</div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </SearchSuggestions>
            )}
          </AnimatePresence>
        </SearchBar>
        
        <MapContainer
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
        >
          <InteractiveWorldMap 
            selectedLocation={selectedLocation}
            onLocationSelect={handleLocationSelect}
            onCoordinatesChange={(coords) => {
              console.log('Coordinates selected:', coords);
              setSearchQuery(`${coords.lat.toFixed(4)}, ${coords.lng.toFixed(4)}`);
            }}
          />
        </MapContainer>
      </MapSection>

      <RightPanel>
        {/* Debug: Test element */}
        <div style={{ 
          background: 'rgba(255, 255, 255, 0.2)', 
          padding: '1rem', 
          borderRadius: '8px',
          color: '#fff',
          fontSize: '1.2rem'
        }}>
          🌤️ Weather Panel - Test
        </div>
        
        <WeatherCard>
          <div className="location">
            <FaMapMarkerAlt className="location-icon" />
            <span>{selectedLocation.name}</span>
          </div>
          
          <div className="current-weather">
            <div className="weather-icon">{currentWeather.icon}</div>
            <div className="temperature">{currentWeather.temperature}°C</div>
            <div className="condition">{currentWeather.condition}</div>
          </div>
          
          <div className="weather-details">
            <div className="detail-item">
              <FaThermometerHalf className="detail-icon" />
              <div className="detail-label">Feels like</div>
              <div className="detail-value">{currentWeather.feelsLike}°C</div>
            </div>
            <div className="detail-item">
              <FaWind className="detail-icon" />
              <div className="detail-label">Wind</div>
              <div className="detail-value">{currentWeather.windSpeed} km/h {currentWeather.windDirection}</div>
            </div>
            <div className="detail-item">
              <FaCloudRain className="detail-icon" />
              <div className="detail-label">Humidity</div>
              <div className="detail-value">{currentWeather.humidity}%</div>
            </div>
            <div className="detail-item">
              <FaCloudRain className="detail-icon" />
              <div className="detail-label">Rain Chance</div>
              <div className="detail-value">{currentWeather.precipitation}%</div>
            </div>
          </div>
        </WeatherCard>

        <PredictionPanel>
          <h3>
            <FaClock className="prediction-icon" />
            Prediction Query
            <span style={{
              marginLeft: 'auto',
              fontSize: '0.8rem',
              padding: '0.3rem 0.8rem',
              borderRadius: '12px',
              background: apiStatus === 'online' ? '#10b981' : apiStatus === 'offline' ? '#ef4444' : '#f59e0b',
              color: 'white'
            }}>
              {apiStatus === 'online' && '🟢 API Online'}
              {apiStatus === 'offline' && '🔴 API Offline'}
              {apiStatus === 'checking' && '🟡 Checking...'}
            </span>
          </h3>
          <div className="prediction-form">
            <input
              type="date"
              value={predictionQuery.date}
              onChange={(e) => setPredictionQuery({...predictionQuery, date: e.target.value})}
            />
            <input
              type="time"
              value={predictionQuery.time}
              onChange={(e) => setPredictionQuery({...predictionQuery, time: e.target.value})}
            />
            <button 
              onClick={handlePredictionSubmit}
              disabled={isLoading}
            >
              {isLoading ? 'Analyzing...' : 'Generate Prediction'}
            </button>
          </div>
        </PredictionPanel>


        

        
        {predictionResult.isVisible && (
          <AnomalyReport
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="report-header">
              <h3>Anomaly & Disaster Report</h3>
              <div 
                className="threat-badge"
                style={getThreatLevelStyle(predictionResult.threatLevel)}
              >
                {threatLevels[predictionResult.threatLevel]?.icon} {threatLevels[predictionResult.threatLevel]?.label}
              </div>
            </div>
            <div className="report-content">
              <div className="prediction-summary">{predictionResult.summary}</div>
              <div className="details">{showDetails ? predictionResult.details : predictionResult.details?.split('\n').slice(0, 2).join('\n') + '...'}</div>
            </div>
            <div className="report-actions">
              <button 
                className="details-btn"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  console.log('🔥 DETAILS BUTTON CLICKED! Current showDetails:', showDetails);
                  alert('Details button clicked! Toggling details...');
                  setShowDetails(prev => {
                    const newValue = !prev;
                    console.log('🔄 Toggling showDetails from', prev, 'to', newValue);
                    return newValue;
                  });
                }}
                style={{
                  background: 'linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  padding: '0.8rem 1.5rem',
                  cursor: 'pointer',
                  fontWeight: '500',
                  fontSize: '0.9rem',
                  transition: 'all 0.3s ease',
                  zIndex: '10',
                  position: 'relative'
                }}
                type="button"
              >
                {showDetails ? '📄 Hide Details' : '📋 Show Details'}
              </button>
              <button 
                className="track-btn"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  console.log('🔥 TRACK BUTTON CLICKED! Current isTracking:', isTracking);
                  alert('Track button clicked!');
                  setIsTracking(prev => {
                    const newState = !prev;
                    console.log('🔄 Toggling isTracking from', prev, 'to', newState);
                    if (newState) {
                      console.log('🔍 Started tracking anomaly for location:', selectedLocation?.name);
                      setTimeout(() => alert(`🔍 Started tracking weather alerts for ${selectedLocation?.name || 'selected location'}`), 100);
                    } else {
                      console.log('⏹️ Stopped tracking anomaly');
                      setTimeout(() => alert('⏹️ Stopped tracking weather alerts'), 100);
                    }
                    return newState;
                  });
                }}
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  color: 'white',
                  borderRadius: '8px',
                  padding: '0.8rem 1.5rem',
                  cursor: 'pointer',
                  fontWeight: '500',
                  fontSize: '0.9rem',
                  transition: 'all 0.3s ease',
                  zIndex: '10',
                  position: 'relative'
                }}
                type="button"
              >
                {isTracking ? '⏹️ Stop Tracking' : '📍 Track Alert'}
              </button>
            </div>
          </AnomalyReport>
        )}
      </RightPanel>
    </DashboardContainer>
  );
}

export default Dashboard;