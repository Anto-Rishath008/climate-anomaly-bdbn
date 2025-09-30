// Climate Database Service - Comprehensive Database Operations
import { supabase, TABLES, CHANNELS } from '../config/supabase';

class ClimateDbService {
  constructor() {
    this.realTimeSubscriptions = [];
  }

  // ==================== WEATHER STATIONS ====================
  
  async getWeatherStations() {
    try {
      const { data, error } = await supabase
        .from(TABLES.WEATHER_STATIONS)
        .select('*')
        .order('name');
      
      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error fetching weather stations:', error);
      return [];
    }
  }

  async createWeatherStation(stationData) {
    try {
      const { data, error } = await supabase
        .from(TABLES.WEATHER_STATIONS)
        .insert([{
          name: stationData.name,
          latitude: stationData.latitude,
          longitude: stationData.longitude,
          elevation: stationData.elevation,
          country: stationData.country,
          region: stationData.region,
          status: 'active',
          created_at: new Date().toISOString()
        }])
        .select();

      if (error) throw error;
      return data[0];
    } catch (error) {
      console.error('Error creating weather station:', error);
      throw error;
    }
  }

  // ==================== CLIMATE ANOMALIES ====================
  
  async getClimateAnomalies(limit = 100, filters = {}) {
    try {
      let query = supabase
        .from(TABLES.CLIMATE_ANOMALIES)
        .select(`
          *,
          weather_stations(name, latitude, longitude)
        `)
        .order('detected_at', { ascending: false })
        .limit(limit);

      // Apply filters
      if (filters.anomalyType) {
        query = query.eq('anomaly_type', filters.anomalyType);
      }
      if (filters.severity) {
        query = query.eq('severity', filters.severity);
      }
      if (filters.dateFrom) {
        query = query.gte('detected_at', filters.dateFrom);
      }
      if (filters.dateTo) {
        query = query.lte('detected_at', filters.dateTo);
      }

      const { data, error } = await query;
      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error fetching climate anomalies:', error);
      return [];
    }
  }

  async createClimateAnomaly(anomalyData) {
    try {
      const { data, error } = await supabase
        .from(TABLES.CLIMATE_ANOMALIES)
        .insert([{
          station_id: anomalyData.stationId,
          anomaly_type: anomalyData.type,
          severity: anomalyData.severity,
          confidence_score: anomalyData.confidence,
          temperature_anomaly: anomalyData.temperature,
          precipitation_anomaly: anomalyData.precipitation,
          pressure_anomaly: anomalyData.pressure,
          wind_anomaly: anomalyData.wind,
          humidity_anomaly: anomalyData.humidity,
          detected_at: anomalyData.detectedAt || new Date().toISOString(),
          metadata: anomalyData.metadata || {}
        }])
        .select();

      if (error) throw error;
      return data[0];
    } catch (error) {
      console.error('Error creating climate anomaly:', error);
      throw error;
    }
  }

  // ==================== HISTORICAL DATA ====================
  
  async getHistoricalData(stationId, dateRange) {
    try {
      const { data, error } = await supabase
        .from(TABLES.HISTORICAL_DATA)
        .select('*')
        .eq('station_id', stationId)
        .gte('date', dateRange.from)
        .lte('date', dateRange.to)
        .order('date', { ascending: true });

      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error fetching historical data:', error);
      return [];
    }
  }

  async bulkInsertHistoricalData(historicalData) {
    try {
      const batchSize = 1000;
      const batches = [];
      
      for (let i = 0; i < historicalData.length; i += batchSize) {
        batches.push(historicalData.slice(i, i + batchSize));
      }

      const results = [];
      for (const batch of batches) {
        const { data, error } = await supabase
          .from(TABLES.HISTORICAL_DATA)
          .insert(batch)
          .select();
          
        if (error) throw error;
        results.push(...data);
      }

      return results;
    } catch (error) {
      console.error('Error bulk inserting historical data:', error);
      throw error;
    }
  }

  // ==================== MODEL PREDICTIONS ====================
  
  async saveModelPrediction(predictionData) {
    try {
      const { data, error } = await supabase
        .from(TABLES.MODEL_PREDICTIONS)
        .insert([{
          model_type: predictionData.modelType,
          model_version: predictionData.modelVersion,
          station_id: predictionData.stationId,
          prediction_date: predictionData.predictionDate,
          predicted_values: predictionData.predictions,
          confidence_intervals: predictionData.confidenceIntervals,
          uncertainty_metrics: predictionData.uncertaintyMetrics,
          bayesian_parameters: predictionData.bayesianParams,
          created_at: new Date().toISOString()
        }])
        .select();

      if (error) throw error;
      return data[0];
    } catch (error) {
      console.error('Error saving model prediction:', error);
      throw error;
    }
  }

  async getModelPredictions(filters = {}) {
    try {
      let query = supabase
        .from(TABLES.MODEL_PREDICTIONS)
        .select(`
          *,
          weather_stations(name, latitude, longitude)
        `)
        .order('created_at', { ascending: false });

      if (filters.modelType) {
        query = query.eq('model_type', filters.modelType);
      }
      if (filters.stationId) {
        query = query.eq('station_id', filters.stationId);
      }
      if (filters.limit) {
        query = query.limit(filters.limit);
      }

      const { data, error } = await query;
      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error fetching model predictions:', error);
      return [];
    }
  }

  // ==================== ANOMALY ALERTS ====================
  
  async createAnomalyAlert(alertData) {
    try {
      const { data, error } = await supabase
        .from(TABLES.ANOMALY_ALERTS)
        .insert([{
          anomaly_id: alertData.anomalyId,
          alert_level: alertData.level,
          message: alertData.message,
          affected_regions: alertData.affectedRegions,
          recommended_actions: alertData.recommendedActions,
          status: 'active',
          created_at: new Date().toISOString()
        }])
        .select();

      if (error) throw error;
      
      // Trigger real-time notification
      await this.broadcastAnomalyAlert(data[0]);
      
      return data[0];
    } catch (error) {
      console.error('Error creating anomaly alert:', error);
      throw error;
    }
  }

  async getActiveAlerts() {
    try {
      const { data, error } = await supabase
        .from(TABLES.ANOMALY_ALERTS)
        .select(`
          *,
          climate_anomalies(anomaly_type, severity, confidence_score),
          weather_stations(name, latitude, longitude)
        `)
        .eq('status', 'active')
        .order('created_at', { ascending: false });

      if (error) throw error;
      return data;
    } catch (error) {
      console.error('Error fetching active alerts:', error);
      return [];
    }
  }

  // ==================== REAL-TIME SUBSCRIPTIONS ====================
  
  subscribeToAnomalyAlerts(callback) {
    const subscription = supabase
      .channel(CHANNELS.ANOMALY_ALERTS)
      .on('postgres_changes', 
        { 
          event: 'INSERT', 
          schema: 'public', 
          table: TABLES.ANOMALY_ALERTS 
        }, 
        callback
      )
      .subscribe();

    this.realTimeSubscriptions.push(subscription);
    return subscription;
  }

  subscribeToWeatherUpdates(callback) {
    const subscription = supabase
      .channel(CHANNELS.WEATHER_UPDATES)
      .on('postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: TABLES.HISTORICAL_DATA
        },
        callback
      )
      .subscribe();

    this.realTimeSubscriptions.push(subscription);
    return subscription;
  }

  subscribeToModelPredictions(callback) {
    const subscription = supabase
      .channel(CHANNELS.MODEL_PREDICTIONS)
      .on('postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: TABLES.MODEL_PREDICTIONS
        },
        callback
      )
      .subscribe();

    this.realTimeSubscriptions.push(subscription);
    return subscription;
  }

  async broadcastAnomalyAlert(alert) {
    try {
      await supabase
        .channel(CHANNELS.ANOMALY_ALERTS)
        .send({
          type: 'broadcast',
          event: 'new_anomaly_alert',
          payload: alert
        });
    } catch (error) {
      console.error('Error broadcasting anomaly alert:', error);
    }
  }

  // ==================== UTILITY METHODS ====================
  
  async getClimateStatistics() {
    try {
      const [anomalies, predictions, alerts] = await Promise.all([
        supabase.from(TABLES.CLIMATE_ANOMALIES).select('*', { count: 'exact', head: true }),
        supabase.from(TABLES.MODEL_PREDICTIONS).select('*', { count: 'exact', head: true }),
        supabase.from(TABLES.ANOMALY_ALERTS).select('*', { count: 'exact', head: true }).eq('status', 'active')
      ]);

      return {
        totalAnomalies: anomalies.count || 0,
        totalPredictions: predictions.count || 0,
        activeAlerts: alerts.count || 0
      };
    } catch (error) {
      console.error('Error fetching climate statistics:', error);
      return { totalAnomalies: 0, totalPredictions: 0, activeAlerts: 0 };
    }
  }

  unsubscribeAll() {
    this.realTimeSubscriptions.forEach(subscription => {
      supabase.removeChannel(subscription);
    });
    this.realTimeSubscriptions = [];
  }

  // ==================== CLEANUP ====================
  
  destroy() {
    this.unsubscribeAll();
  }
}

// Create singleton instance
const climateDbService = new ClimateDbService();
export default climateDbService;