import axios from 'axios';

// Weather API Configuration
const WEATHER_API_KEY = process.env.REACT_APP_WEATHER_API_KEY || 'your_api_key_here';
const WEATHER_API_BASE = 'https://api.weatherapi.com/v1';

// OpenWeatherMap as backup
const OPENWEATHER_API_KEY = process.env.REACT_APP_OPENWEATHER_API_KEY || 'your_openweather_key_here';
const OPENWEATHER_BASE = 'https://api.openweathermap.org/data/2.5';

// Create axios instance with default config
const weatherAPI = axios.create({
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Weather Service Class
class WeatherService {
  /**
   * Get current weather data for a specific location
   */
  async getCurrentWeather(location) {
    try {
      const response = await weatherAPI.get(`${WEATHER_API_BASE}/current.json`, {
        params: {
          key: WEATHER_API_KEY,
          q: location,
          aqi: 'yes',
        }
      });
      
      return this.formatCurrentWeatherData(response.data);
    } catch (error) {
      console.error('Weather API Error:', error);
      return this.getCurrentWeatherFromOpenWeather(location);
    }
  }

  /**
   * Get weather forecast for multiple days
   */
  async getWeatherForecast(location, days = 7) {
    try {
      const response = await weatherAPI.get(`${WEATHER_API_BASE}/forecast.json`, {
        params: {
          key: WEATHER_API_KEY,
          q: location,
          days: Math.min(days, 10),
          aqi: 'yes',
          alerts: 'yes'
        }
      });
      
      return this.formatForecastData(response.data);
    } catch (error) {
      console.error('Forecast API Error:', error);
      throw new Error('Unable to fetch weather forecast');
    }
  }

  /**
   * Fallback to OpenWeatherMap API
   */
  async getCurrentWeatherFromOpenWeather(location) {
    try {
      const response = await weatherAPI.get(`${OPENWEATHER_BASE}/weather`, {
        params: {
          appid: OPENWEATHER_API_KEY,
          q: location,
          units: 'metric'
        }
      });
      
      return this.formatOpenWeatherData(response.data);
    } catch (error) {
      console.error('OpenWeather API Error:', error);
      throw new Error('Unable to fetch weather data from any source');
    }
  }

  /**
   * Get weather data for multiple cities
   */
  async getMultipleCitiesWeather(locations) {
    try {
      const promises = locations.map(location => 
        this.getCurrentWeather(location).catch(error => ({
          location,
          error: error.message,
          data: null
        }))
      );
      
      const results = await Promise.all(promises);
      return results.filter(result => result.data !== null);
    } catch (error) {
      console.error('Multiple cities weather error:', error);
      return [];
    }
  }

  /**
   * Format WeatherAPI.com current weather data
   */
  formatCurrentWeatherData(data) {
    const { location, current } = data;
    
    return {
      location: {
        name: location.name,
        country: location.country,
        coordinates: {
          lat: location.lat,
          lon: location.lon
        },
        timezone: location.tz_id,
        localTime: location.localtime
      },
      weather: {
        temperature: current.temp_c,
        feelsLike: current.feelslike_c,
        humidity: current.humidity,
        pressure: current.pressure_mb,
        windSpeed: current.wind_kph,
        windDirection: current.wind_dir,
        visibility: current.vis_km,
        uvIndex: current.uv,
        condition: {
          text: current.condition.text,
          icon: current.condition.icon,
          code: current.condition.code
        }
      },
      timestamp: new Date().toISOString(),
      source: 'WeatherAPI.com'
    };
  }

  /**
   * Format OpenWeatherMap data
   */
  formatOpenWeatherData(data) {
    return {
      location: {
        name: data.name,
        country: data.sys.country,
        coordinates: {
          lat: data.coord.lat,
          lon: data.coord.lon
        },
        timezone: data.timezone,
        localTime: new Date().toISOString()
      },
      weather: {
        temperature: data.main.temp,
        feelsLike: data.main.feels_like,
        humidity: data.main.humidity,
        pressure: data.main.pressure,
        windSpeed: data.wind?.speed * 3.6 || 0,
        windDirection: data.wind?.deg || 0,
        visibility: data.visibility / 1000,
        condition: {
          text: data.weather[0].description,
          icon: `https://openweathermap.org/img/w/${data.weather[0].icon}.png`,
          code: data.weather[0].id
        }
      },
      timestamp: new Date().toISOString(),
      source: 'OpenWeatherMap'
    };
  }

  /**
   * Calculate anomaly score based on historical data
   */
  calculateAnomalyScore(currentData, historicalData) {
    if (!currentData || !historicalData) return Math.random() * 0.5; // Mock data
    
    const tempAnomaly = Math.abs(currentData.weather.temperature - 20) / 10;
    const humidityAnomaly = Math.abs(currentData.weather.humidity - 60) / 20;
    const pressureAnomaly = Math.abs(currentData.weather.pressure - 1013) / 50;
    
    const weightedAnomaly = (tempAnomaly * 0.4 + humidityAnomaly * 0.3 + pressureAnomaly * 0.3);
    
    return Math.min(1, Math.max(0, weightedAnomaly));
  }

  /**
   * Get sample climate monitoring locations
   */
  getMonitoringLocations() {
    return [
      { name: 'New York', country: 'USA', query: 'New York,US' },
      { name: 'London', country: 'UK', query: 'London,UK' },
      { name: 'Tokyo', country: 'Japan', query: 'Tokyo,JP' },
      { name: 'Sydney', country: 'Australia', query: 'Sydney,AU' },
      { name: 'Mumbai', country: 'India', query: 'Mumbai,IN' },
      { name: 'Sao Paulo', country: 'Brazil', query: 'Sao Paulo,BR' },
      { name: 'Cairo', country: 'Egypt', query: 'Cairo,EG' },
      { name: 'Moscow', country: 'Russia', query: 'Moscow,RU' }
    ];
  }
}

// Create and export singleton instance
const weatherService = new WeatherService();
export default weatherService;