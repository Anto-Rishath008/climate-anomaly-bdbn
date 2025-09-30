-- Climate Anomaly Detection System - Database Schema
-- Run this in your Supabase SQL Editor to create all necessary tables

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- ==================== WEATHER STATIONS TABLE ====================
CREATE TABLE IF NOT EXISTS weather_stations (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name VARCHAR(255) NOT NULL,
  latitude DECIMAL(10, 8) NOT NULL,
  longitude DECIMAL(11, 8) NOT NULL,
  elevation INTEGER,
  country VARCHAR(100),
  region VARCHAR(100),
  status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'maintenance')),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for weather stations
CREATE INDEX IF NOT EXISTS idx_weather_stations_location ON weather_stations (latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_weather_stations_country ON weather_stations (country);
CREATE INDEX IF NOT EXISTS idx_weather_stations_status ON weather_stations (status);

-- ==================== CLIMATE ANOMALIES TABLE ====================
CREATE TABLE IF NOT EXISTS climate_anomalies (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  station_id UUID REFERENCES weather_stations(id) ON DELETE CASCADE,
  anomaly_type VARCHAR(50) NOT NULL CHECK (anomaly_type IN (
    'temperature_spike', 'temperature_drop', 'heatwave', 'cold_wave',
    'drought', 'flooding', 'extreme_precipitation', 'wind_anomaly',
    'pressure_anomaly', 'humidity_anomaly', 'seasonal_shift',
    'multi_variable_anomaly', 'compound_event'
  )),
  severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'extreme')),
  confidence_score DECIMAL(5, 4) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
  temperature_anomaly DECIMAL(8, 4),
  precipitation_anomaly DECIMAL(8, 4),
  pressure_anomaly DECIMAL(8, 4),
  wind_anomaly DECIMAL(8, 4),
  humidity_anomaly DECIMAL(8, 4),
  detected_at TIMESTAMP WITH TIME ZONE NOT NULL,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for climate anomalies
CREATE INDEX IF NOT EXISTS idx_climate_anomalies_station ON climate_anomalies (station_id);
CREATE INDEX IF NOT EXISTS idx_climate_anomalies_type ON climate_anomalies (anomaly_type);
CREATE INDEX IF NOT EXISTS idx_climate_anomalies_severity ON climate_anomalies (severity);
CREATE INDEX IF NOT EXISTS idx_climate_anomalies_detected_at ON climate_anomalies (detected_at);
CREATE INDEX IF NOT EXISTS idx_climate_anomalies_confidence ON climate_anomalies (confidence_score);

-- ==================== HISTORICAL CLIMATE DATA TABLE ====================
CREATE TABLE IF NOT EXISTS historical_climate_data (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  station_id UUID REFERENCES weather_stations(id) ON DELETE CASCADE,
  date DATE NOT NULL,
  temperature_max DECIMAL(6, 3),
  temperature_min DECIMAL(6, 3),
  temperature_avg DECIMAL(6, 3),
  precipitation DECIMAL(8, 4),
  humidity DECIMAL(5, 2),
  pressure DECIMAL(8, 2),
  wind_speed DECIMAL(6, 3),
  wind_direction INTEGER,
  cloud_cover DECIMAL(5, 2),
  visibility DECIMAL(6, 3),
  uv_index DECIMAL(4, 2),
  solar_radiation DECIMAL(8, 4),
  evapotranspiration DECIMAL(8, 4),
  soil_temperature DECIMAL(6, 3),
  soil_moisture DECIMAL(5, 2),
  data_quality_score DECIMAL(3, 2) DEFAULT 1.0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for historical data
CREATE INDEX IF NOT EXISTS idx_historical_data_station_date ON historical_climate_data (station_id, date);
CREATE INDEX IF NOT EXISTS idx_historical_data_date ON historical_climate_data (date);
CREATE UNIQUE INDEX IF NOT EXISTS idx_historical_data_unique ON historical_climate_data (station_id, date);

-- ==================== MODEL PREDICTIONS TABLE ====================
CREATE TABLE IF NOT EXISTS model_predictions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  model_type VARCHAR(50) NOT NULL CHECK (model_type IN (
    'bayesian_dbn', 'conv_nll', 'unet_small', 'ensemble', 'lstm_anomaly'
  )),
  model_version VARCHAR(20) NOT NULL,
  station_id UUID REFERENCES weather_stations(id) ON DELETE CASCADE,
  prediction_date DATE NOT NULL,
  horizon_days INTEGER NOT NULL DEFAULT 7,
  predicted_values JSONB NOT NULL,
  confidence_intervals JSONB,
  uncertainty_metrics JSONB,
  bayesian_parameters JSONB,
  accuracy_score DECIMAL(5, 4),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for model predictions
CREATE INDEX IF NOT EXISTS idx_model_predictions_station ON model_predictions (station_id);
CREATE INDEX IF NOT EXISTS idx_model_predictions_type ON model_predictions (model_type);
CREATE INDEX IF NOT EXISTS idx_model_predictions_date ON model_predictions (prediction_date);
CREATE INDEX IF NOT EXISTS idx_model_predictions_created ON model_predictions (created_at);

-- ==================== ANOMALY ALERTS TABLE ====================
CREATE TABLE IF NOT EXISTS anomaly_alerts (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  anomaly_id UUID REFERENCES climate_anomalies(id) ON DELETE CASCADE,
  alert_level VARCHAR(20) NOT NULL CHECK (alert_level IN ('info', 'warning', 'critical', 'emergency')),
  title VARCHAR(255) NOT NULL,
  message TEXT NOT NULL,
  affected_regions TEXT[],
  recommended_actions TEXT[],
  status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'acknowledged', 'resolved')),
  acknowledged_by VARCHAR(255),
  acknowledged_at TIMESTAMP WITH TIME ZONE,
  resolved_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for anomaly alerts
CREATE INDEX IF NOT EXISTS idx_anomaly_alerts_anomaly ON anomaly_alerts (anomaly_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_alerts_level ON anomaly_alerts (alert_level);
CREATE INDEX IF NOT EXISTS idx_anomaly_alerts_status ON anomaly_alerts (status);
CREATE INDEX IF NOT EXISTS idx_anomaly_alerts_created ON anomaly_alerts (created_at);

-- ==================== USER SESSIONS TABLE ====================
CREATE TABLE IF NOT EXISTS user_sessions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID,
  session_token VARCHAR(255) UNIQUE NOT NULL,
  ip_address INET,
  user_agent TEXT,
  location JSONB,
  preferences JSONB DEFAULT '{}',
  last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for user sessions
CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions (session_token);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires ON user_sessions (expires_at);

-- ==================== SYSTEM LOGS TABLE ====================
CREATE TABLE IF NOT EXISTS system_logs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  log_level VARCHAR(20) NOT NULL CHECK (log_level IN ('debug', 'info', 'warning', 'error', 'critical')),
  component VARCHAR(50) NOT NULL,
  message TEXT NOT NULL,
  details JSONB,
  user_id UUID,
  session_id UUID REFERENCES user_sessions(id) ON DELETE SET NULL,
  ip_address INET,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for system logs
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs (log_level);
CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs (component);
CREATE INDEX IF NOT EXISTS idx_system_logs_created ON system_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_system_logs_user ON system_logs (user_id);

-- ==================== TRIGGERS FOR UPDATED_AT ====================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
CREATE TRIGGER update_weather_stations_updated_at BEFORE UPDATE ON weather_stations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_anomaly_alerts_updated_at BEFORE UPDATE ON anomaly_alerts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==================== ROW LEVEL SECURITY (RLS) ====================
-- Enable RLS on all tables
ALTER TABLE weather_stations ENABLE ROW LEVEL SECURITY;
ALTER TABLE climate_anomalies ENABLE ROW LEVEL SECURITY;
ALTER TABLE historical_climate_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE anomaly_alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_logs ENABLE ROW LEVEL SECURITY;

-- Create policies for public read access (you can modify these based on your security needs)
CREATE POLICY "Public read access for weather stations" ON weather_stations FOR SELECT USING (true);
CREATE POLICY "Public read access for climate anomalies" ON climate_anomalies FOR SELECT USING (true);
CREATE POLICY "Public read access for historical data" ON historical_climate_data FOR SELECT USING (true);
CREATE POLICY "Public read access for model predictions" ON model_predictions FOR SELECT USING (true);
CREATE POLICY "Public read access for anomaly alerts" ON anomaly_alerts FOR SELECT USING (true);

-- Create policies for authenticated insert/update (modify based on your auth setup)
CREATE POLICY "Authenticated insert for climate anomalies" ON climate_anomalies FOR INSERT WITH CHECK (true);
CREATE POLICY "Authenticated insert for historical data" ON historical_climate_data FOR INSERT WITH CHECK (true);
CREATE POLICY "Authenticated insert for model predictions" ON model_predictions FOR INSERT WITH CHECK (true);
CREATE POLICY "Authenticated insert for anomaly alerts" ON anomaly_alerts FOR INSERT WITH CHECK (true);

-- ==================== SAMPLE DATA ====================
-- Insert some sample weather stations
INSERT INTO weather_stations (name, latitude, longitude, elevation, country, region) VALUES
('New York Central Park', 40.7829, -73.9654, 42, 'USA', 'New York'),
('London Heathrow', 51.4700, -0.4543, 25, 'UK', 'England'),
('Tokyo Haneda', 35.5494, 139.7798, 6, 'Japan', 'Kanto'),
('Sydney Observatory', -33.8688, 151.2093, 39, 'Australia', 'New South Wales'),
('Mumbai Santacruz', 19.0896, 72.8656, 11, 'India', 'Maharashtra'),
('Berlin Tempelhof', 52.4675, 13.4021, 48, 'Germany', 'Berlin'),
('São Paulo Congonhas', -23.6262, -46.6554, 803, 'Brazil', 'São Paulo'),
('Cairo International', 30.1219, 31.4056, 74, 'Egypt', 'Cairo')
ON CONFLICT DO NOTHING;

-- Create a view for climate anomaly statistics
CREATE OR REPLACE VIEW climate_anomaly_stats AS
SELECT 
  ws.name as station_name,
  ws.country,
  ca.anomaly_type,
  ca.severity,
  COUNT(*) as anomaly_count,
  AVG(ca.confidence_score) as avg_confidence,
  MAX(ca.detected_at) as last_detected
FROM climate_anomalies ca
JOIN weather_stations ws ON ca.station_id = ws.id
GROUP BY ws.name, ws.country, ca.anomaly_type, ca.severity
ORDER BY anomaly_count DESC;

-- Create a view for recent predictions
CREATE OR REPLACE VIEW recent_predictions AS
SELECT 
  mp.*,
  ws.name as station_name,
  ws.latitude,
  ws.longitude,
  ws.country
FROM model_predictions mp
JOIN weather_stations ws ON mp.station_id = ws.id
WHERE mp.created_at >= NOW() - INTERVAL '7 days'
ORDER BY mp.created_at DESC;

COMMENT ON TABLE weather_stations IS 'Weather monitoring stations worldwide';
COMMENT ON TABLE climate_anomalies IS 'Detected climate anomalies with Bayesian confidence scores';
COMMENT ON TABLE historical_climate_data IS 'Historical weather and climate measurements';
COMMENT ON TABLE model_predictions IS 'ML model predictions with uncertainty quantification';
COMMENT ON TABLE anomaly_alerts IS 'System-generated alerts for climate anomalies';
COMMENT ON TABLE user_sessions IS 'User session management and preferences';
COMMENT ON TABLE system_logs IS 'Application logging and audit trail';