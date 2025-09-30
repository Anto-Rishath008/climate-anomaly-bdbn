/**
 * Climate ML API Service
 * Handles communication with the Flask ML backend
 */

class ClimateMLApiService {
  constructor() {
    this.baseUrl = 'http://localhost:5000/api';
    this.isConnected = false;
  }

  /**
   * Check if the ML API is available
   */
  async checkHealth() {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        this.isConnected = true;
        return {
          success: true,
          status: data.status,
          modelsLoaded: data.models_loaded || [],
          timestamp: data.timestamp
        };
      } else {
        this.isConnected = false;
        return {
          success: false,
          error: `HTTP ${response.status}: ${response.statusText}`
        };
      }
    } catch (error) {
      this.isConnected = false;
      return {
        success: false,
        error: error.message || 'Connection failed'
      };
    }
  }

  /**
   * Get information about loaded models
   */
  async getModelInfo() {
    try {
      const response = await fetch(`${this.baseUrl}/models/info`);
      
      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          models: data.loaded_models,
          details: data.model_details
        };
      } else {
        return {
          success: false,
          error: `HTTP ${response.status}: ${response.statusText}`
        };
      }
    } catch (error) {
      return {
        success: false,
        error: error.message || 'Failed to fetch model info'
      };
    }
  }

  /**
   * Make a general prediction with uncertainty quantification
   */
  async makePrediction(modelName, inputData, nSamples = 100) {
    try {
      const response = await fetch(`${this.baseUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: modelName,
          input_data: inputData,
          n_samples: nSamples
        })
      });

      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          prediction: data.prediction,
          uncertainty: data.uncertainty,
          confidenceInterval: data.confidence_interval,
          predictionType: data.prediction_type,
          modelUsed: data.model_used,
          probabilityConcepts: data.probability_concepts,
          timestamp: data.timestamp
        };
      } else {
        const errorData = await response.json();
        return {
          success: false,
          error: errorData.error || `HTTP ${response.status}: ${response.statusText}`
        };
      }
    } catch (error) {
      return {
        success: false,
        error: error.message || 'Prediction failed'
      };
    }
  }

  /**
   * Make a climate anomaly prediction
   */
  async predictClimateAnomaly(climateData) {
    try {
      const response = await fetch(`${this.baseUrl}/predict/climate-anomaly`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(climateData)
      });

      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          isAnomaly: data.is_anomaly,
          anomalyScore: data.anomaly_score,
          confidence: data.confidence,
          models: data.models,
          inputParameters: data.input_parameters,
          probabilityAnalysis: data.probability_analysis,
          timestamp: data.timestamp
        };
      } else {
        const errorData = await response.json();
        return {
          success: false,
          error: errorData.error || `HTTP ${response.status}: ${response.statusText}`
        };
      }
    } catch (error) {
      return {
        success: false,
        error: error.message || 'Climate anomaly prediction failed'
      };
    }
  }

  /**
   * Get simulated historical predictions
   */
  async getHistoricalPredictions(days = 30) {
    try {
      const response = await fetch(`${this.baseUrl}/simulate/historical`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ days })
      });

      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          historicalData: data.historical_predictions,
          summary: data.summary
        };
      } else {
        const errorData = await response.json();
        return {
          success: false,
          error: errorData.error || `HTTP ${response.status}: ${response.statusText}`
        };
      }
    } catch (error) {
      return {
        success: false,
        error: error.message || 'Historical data simulation failed'
      };
    }
  }

  /**
   * Generate realistic climate parameters for testing
   */
  generateRandomClimateData() {
    return {
      temperature: 15 + Math.random() * 25, // 15-40°C
      humidity: 30 + Math.random() * 50,    // 30-80%
      pressure: 980 + Math.random() * 60,   // 980-1040 hPa
      wind_speed: Math.random() * 25,       // 0-25 m/s
      precipitation: Math.random() * 15     // 0-15 mm
    };
  }

  /**
   * Generate extreme weather conditions for anomaly testing
   */
  generateExtremeClimateData() {
    const scenarios = [
      // Heatwave
      {
        temperature: 45 + Math.random() * 10,
        humidity: 20 + Math.random() * 20,
        pressure: 1000 + Math.random() * 20,
        wind_speed: Math.random() * 10,
        precipitation: 0
      },
      // Storm
      {
        temperature: 20 + Math.random() * 10,
        humidity: 80 + Math.random() * 15,
        pressure: 960 + Math.random() * 20,
        wind_speed: 30 + Math.random() * 20,
        precipitation: 20 + Math.random() * 30
      },
      // Cold snap
      {
        temperature: -10 + Math.random() * 10,
        humidity: 70 + Math.random() * 20,
        pressure: 1030 + Math.random() * 20,
        wind_speed: 15 + Math.random() * 15,
        precipitation: Math.random() * 5
      },
      // Drought
      {
        temperature: 35 + Math.random() * 10,
        humidity: 10 + Math.random() * 20,
        pressure: 1015 + Math.random() * 15,
        wind_speed: Math.random() * 15,
        precipitation: 0
      }
    ];

    return scenarios[Math.floor(Math.random() * scenarios.length)];
  }

  /**
   * Get connection status
   */
  getConnectionStatus() {
    return this.isConnected;
  }
}

// Export singleton instance
const climateMLApi = new ClimateMLApiService();
export default climateMLApi;