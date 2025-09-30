// Database Initialization Script
// Run this after setting up your Supabase schema to populate with sample data

import climateDbService from '../services/climateDbService.js';
import { supabase } from '../config/supabase.js';

class DatabaseInitializer {
  constructor() {
    this.sampleData = {
      anomalies: [
        {
          stationId: null, // Will be set from weather stations
          type: 'temperature_spike',
          severity: 'high',
          confidence: 0.95,
          temperature: 8.5,
          precipitation: 0,
          pressure: -5.2,
          wind: 2.1,
          humidity: -15.3,
          detectedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(), // 2 days ago
          metadata: {
            duration_hours: 18,
            affected_area_km2: 1250,
            model_used: 'bayesian_dbn_v2.1'
          }
        },
        {
          stationId: null,
          type: 'extreme_precipitation',
          severity: 'extreme',
          confidence: 0.92,
          temperature: -2.1,
          precipitation: 45.8,
          pressure: -8.7,
          wind: 12.4,
          humidity: 25.6,
          detectedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(), // 1 day ago
          metadata: {
            duration_hours: 36,
            affected_area_km2: 3500,
            model_used: 'conv_nll_v1.8'
          }
        },
        {
          stationId: null,
          type: 'drought',
          severity: 'medium',
          confidence: 0.87,
          temperature: 4.2,
          precipitation: -85.4,
          pressure: 3.1,
          wind: -1.8,
          humidity: -42.7,
          detectedAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(), // 5 days ago
          metadata: {
            duration_days: 45,
            affected_area_km2: 15000,
            model_used: 'ensemble_v3.2'
          }
        }
      ],
      historicalData: [
        // Will generate synthetic historical data for past 30 days
      ],
      predictions: [
        {
          modelType: 'bayesian_dbn',
          modelVersion: 'v2.1',
          stationId: null,
          predictionDate: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0], // 7 days from now
          predictions: {
            temperature: { mean: 22.5, std: 2.1, min: 18.2, max: 26.8 },
            precipitation: { mean: 2.4, std: 1.8, min: 0, max: 8.5 },
            humidity: { mean: 65.2, std: 8.4, min: 45.3, max: 82.1 },
            pressure: { mean: 1013.2, std: 5.7, min: 1001.4, max: 1025.8 }
          },
          confidenceIntervals: {
            temperature: { lower: 20.4, upper: 24.6, confidence: 0.95 },
            precipitation: { lower: 0.6, upper: 4.2, confidence: 0.95 }
          },
          uncertaintyMetrics: {
            epistemic_uncertainty: 0.15,
            aleatoric_uncertainty: 0.23,
            total_uncertainty: 0.28
          },
          bayesianParams: {
            posterior_samples: 1000,
            mcmc_chains: 4,
            convergence_rhat: 1.02
          }
        }
      ]
    };
  }

  async initialize() {
    try {
      console.log('🚀 Starting database initialization...');
      
      // Step 1: Get weather stations
      const stations = await climateDbService.getWeatherStations();
      if (stations.length === 0) {
        throw new Error('No weather stations found. Please run the SQL schema first.');
      }
      console.log(`✅ Found ${stations.length} weather stations`);

      // Step 2: Generate and insert historical data
      await this.insertHistoricalData(stations);
      
      // Step 3: Insert climate anomalies
      await this.insertClimateAnomalies(stations);
      
      // Step 4: Insert model predictions
      await this.insertModelPredictions(stations);
      
      // Step 5: Create some alerts
      await this.createSampleAlerts();
      
      console.log('🎉 Database initialization completed successfully!');
      return true;
      
    } catch (error) {
      console.error('❌ Database initialization failed:', error);
      throw error;
    }
  }

  async insertHistoricalData(stations) {
    console.log('📊 Generating historical climate data...');
    
    const historicalData = [];
    const daysToGenerate = 90; // Last 90 days
    
    for (const station of stations.slice(0, 3)) { // First 3 stations only for demo
      for (let i = daysToGenerate; i >= 1; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        
        // Generate realistic weather data with some randomness
        const baseTemp = this.getBaseTemperature(station.latitude, date);
        const seasonalVariation = Math.sin((date.getMonth() + 1) * Math.PI / 6) * 10;
        
        historicalData.push({
          station_id: station.id,
          date: date.toISOString().split('T')[0],
          temperature_max: baseTemp + seasonalVariation + (Math.random() * 8 - 4),
          temperature_min: baseTemp + seasonalVariation - 8 + (Math.random() * 4 - 2),
          temperature_avg: baseTemp + seasonalVariation + (Math.random() * 4 - 2),
          precipitation: Math.max(0, Math.random() * 15 - 12), // Mostly dry with occasional rain
          humidity: 50 + (Math.random() * 40 - 20),
          pressure: 1013 + (Math.random() * 30 - 15),
          wind_speed: Math.random() * 20,
          wind_direction: Math.floor(Math.random() * 360),
          cloud_cover: Math.random() * 100,
          visibility: 10 + (Math.random() * 40 - 20),
          uv_index: Math.max(0, Math.min(11, 5 + (Math.random() * 6 - 3))),
          data_quality_score: 0.85 + (Math.random() * 0.15)
        });
      }
    }
    
    // Bulk insert historical data
    await climateDbService.bulkInsertHistoricalData(historicalData);
    console.log(`✅ Inserted ${historicalData.length} historical data records`);
  }

  async insertClimateAnomalies(stations) {
    console.log('🌡️ Creating climate anomalies...');
    
    let createdCount = 0;
    for (const anomaly of this.sampleData.anomalies) {
      // Assign random station
      anomaly.stationId = stations[Math.floor(Math.random() * stations.length)].id;
      
      const createdAnomaly = await climateDbService.createClimateAnomaly(anomaly);
      if (createdAnomaly) {
        createdCount++;
      }
    }
    
    console.log(`✅ Created ${createdCount} climate anomalies`);
  }

  async insertModelPredictions(stations) {
    console.log('🤖 Generating model predictions...');
    
    let createdCount = 0;
    for (const prediction of this.sampleData.predictions) {
      // Create predictions for multiple stations
      for (const station of stations.slice(0, 5)) {
        prediction.stationId = station.id;
        
        const createdPrediction = await climateDbService.saveModelPrediction(prediction);
        if (createdPrediction) {
          createdCount++;
        }
      }
    }
    
    console.log(`✅ Created ${createdCount} model predictions`);
  }

  async createSampleAlerts() {
    console.log('🚨 Creating sample alerts...');
    
    // Get recent anomalies to create alerts for
    const recentAnomalies = await climateDbService.getClimateAnomalies(5);
    
    let alertCount = 0;
    for (const anomaly of recentAnomalies.slice(0, 2)) {
      const alertData = {
        anomalyId: anomaly.id,
        level: this.getAlertLevel(anomaly.severity),
        title: `${anomaly.anomaly_type.replace('_', ' ').toUpperCase()} Alert`,
        message: this.generateAlertMessage(anomaly),
        affectedRegions: [`${anomaly.weather_stations?.name} region`, 'Surrounding areas'],
        recommendedActions: this.getRecommendedActions(anomaly.anomaly_type)
      };
      
      const createdAlert = await climateDbService.createAnomalyAlert(alertData);
      if (createdAlert) {
        alertCount++;
      }
    }
    
    console.log(`✅ Created ${alertCount} anomaly alerts`);
  }

  // Helper methods
  getBaseTemperature(latitude, date) {
    // Simple temperature model based on latitude and season
    const latitudeFactor = Math.cos(latitude * Math.PI / 180) * 30;
    return 15 + latitudeFactor;
  }

  getAlertLevel(severity) {
    const mapping = {
      'low': 'info',
      'medium': 'warning',
      'high': 'critical',
      'extreme': 'emergency'
    };
    return mapping[severity] || 'info';
  }

  generateAlertMessage(anomaly) {
    const typeMessages = {
      'temperature_spike': 'Unusual temperature spike detected',
      'extreme_precipitation': 'Extreme precipitation event in progress',
      'drought': 'Extended drought conditions identified',
      'flooding': 'Potential flooding risk detected',
      'heatwave': 'Heatwave conditions detected',
      'cold_wave': 'Cold wave conditions detected'
    };
    
    return typeMessages[anomaly.anomaly_type] || 'Climate anomaly detected';
  }

  getRecommendedActions(anomalyType) {
    const actionMap = {
      'temperature_spike': [
        'Monitor heat-sensitive infrastructure',
        'Issue heat warnings to vulnerable populations',
        'Check cooling systems capacity'
      ],
      'extreme_precipitation': [
        'Monitor flood-prone areas',
        'Check drainage systems',
        'Issue travel advisories'
      ],
      'drought': [
        'Implement water conservation measures',
        'Monitor agricultural areas',
        'Prepare drought response plans'
      ]
    };
    
    return actionMap[anomalyType] || ['Monitor situation closely', 'Prepare contingency plans'];
  }

  // Test database connection
  async testConnection() {
    try {
      const { data, error } = await supabase
        .from('weather_stations')
        .select('count')
        .limit(1);
        
      if (error) {
        throw error;
      }
      
      console.log('✅ Database connection successful');
      return true;
    } catch (error) {
      console.error('❌ Database connection failed:', error);
      return false;
    }
  }
}

// Create and export initializer instance
const dbInitializer = new DatabaseInitializer();
export default dbInitializer;

// Self-executing initialization (uncomment to run automatically)
// dbInitializer.initialize().catch(console.error);