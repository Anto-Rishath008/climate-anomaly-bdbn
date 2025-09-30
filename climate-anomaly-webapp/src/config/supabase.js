// Supabase Configuration for Climate Anomaly Detection System
import { createClient } from '@supabase/supabase-js';

// Supabase configuration
const supabaseUrl = process.env.REACT_APP_SUPABASE_URL || 'https://awxqkescyeetesyhdtxj.supabase.co';
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF3eHFrZXNjeWVldGVzeWhkdHhqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg4ODgwMDMsImV4cCI6MjA3NDQ2NDAwM30.4fqHNBtmUTKYF0Pt6UUhg1JUdm3-bkgEk8YdeCPyf5U';

// Create Supabase client
export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Database configuration for direct PostgreSQL connection
export const dbConfig = {
  connectionString: process.env.DATABASE_URL || 'postgresql://postgres:nopassword@db.awxqkescyeetesyhdtxj.supabase.co:5432/postgres',
  ssl: {
    rejectUnauthorized: false
  }
};

// Database table names
export const TABLES = {
  WEATHER_STATIONS: 'weather_stations',
  CLIMATE_ANOMALIES: 'climate_anomalies', 
  HISTORICAL_DATA: 'historical_climate_data',
  MODEL_PREDICTIONS: 'model_predictions',
  ANOMALY_ALERTS: 'anomaly_alerts',
  USER_SESSIONS: 'user_sessions',
  SYSTEM_LOGS: 'system_logs'
};

// Real-time channels
export const CHANNELS = {
  ANOMALY_ALERTS: 'anomaly-alerts',
  WEATHER_UPDATES: 'weather-updates',
  MODEL_PREDICTIONS: 'model-predictions'
};

export default supabase;