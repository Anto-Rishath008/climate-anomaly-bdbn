# Supabase Database Setup Guide

## 🚀 Getting Started with Supabase

### Step 1: Access Your Supabase Project
1. Go to [https://supabase.com](https://supabase.com)
2. Sign in to your account
3. Navigate to your project: `awxqkescyeetesyhdtxj`

### Step 2: Get Your Supabase Keys
1. In your Supabase dashboard, go to **Settings** → **API**
2. Copy your **Project URL** and **anon public key**
3. Update your `.env` file:

```env
REACT_APP_SUPABASE_URL=https://awxqkescyeetesyhdtxj.supabase.co
REACT_APP_SUPABASE_ANON_KEY=your_actual_anon_key_here
```

### Step 3: Create Database Schema
1. In Supabase dashboard, go to **SQL Editor**
2. Copy and paste the entire content from `database/schema.sql`
3. Click **Run** to create all tables and indexes

### Step 4: Verify Schema Creation
After running the schema, you should see these tables in your **Table Editor**:
- `weather_stations` (8 sample stations)
- `climate_anomalies`
- `historical_climate_data`
- `model_predictions`
- `anomaly_alerts`
- `user_sessions`
- `system_logs`

## 🔧 Database Features

### Real-time Subscriptions
The system supports real-time updates for:
- **Anomaly Alerts**: Live notifications when new anomalies are detected
- **Weather Updates**: Real-time weather data changes
- **Model Predictions**: Live ML model prediction results

### Security Features
- **Row Level Security (RLS)** enabled on all tables
- **Public read access** for climate data (can be restricted)
- **Authenticated write access** for data insertion
- **Audit logging** for all system operations

### Performance Optimizations
- **Proper indexing** on frequently queried columns
- **Composite indexes** for complex queries
- **JSONB storage** for flexible metadata
- **Partitioning ready** for large datasets

## 📊 Database Tables Overview

### Weather Stations
- **Purpose**: Store monitoring station information
- **Key Fields**: name, latitude, longitude, elevation, country
- **Sample Data**: 8 global weather stations included

### Climate Anomalies
- **Purpose**: Store detected climate anomalies with confidence scores
- **Key Fields**: anomaly_type, severity, confidence_score, various anomaly measurements
- **Supported Types**: temperature_spike, drought, flooding, heatwave, etc.

### Historical Climate Data
- **Purpose**: Store historical weather measurements
- **Key Fields**: temperature, precipitation, humidity, pressure, wind data
- **Features**: Data quality scoring, comprehensive meteorological variables

### Model Predictions
- **Purpose**: Store ML model predictions with uncertainty quantification
- **Key Fields**: predicted_values, confidence_intervals, bayesian_parameters
- **Supported Models**: bayesian_dbn, conv_nll, unet_small, ensemble

### Anomaly Alerts
- **Purpose**: Manage system alerts for climate anomalies
- **Key Fields**: alert_level, message, affected_regions, recommended_actions
- **Status Tracking**: active, acknowledged, resolved

## 🚀 Using the Database Service

### Basic Usage Example

```javascript
import climateDbService from './services/climateDbService';

// Get all weather stations
const stations = await climateDbService.getWeatherStations();

// Create a new climate anomaly
const anomaly = await climateDbService.createClimateAnomaly({
  stationId: stations[0].id,
  type: 'temperature_spike',
  severity: 'high',
  confidence: 0.95,
  temperature: 8.5,
  detectedAt: new Date().toISOString()
});

// Subscribe to real-time anomaly alerts
climateDbService.subscribeToAnomalyAlerts((payload) => {
  console.log('New anomaly alert:', payload);
});
```

### Advanced Features

#### Bulk Data Operations
```javascript
// Bulk insert historical data
const historicalData = [/* array of data */];
await climateDbService.bulkInsertHistoricalData(historicalData);
```

#### Complex Filtering
```javascript
// Get anomalies with filters
const anomalies = await climateDbService.getClimateAnomalies(100, {
  anomalyType: 'temperature_spike',
  severity: 'high',
  dateFrom: '2024-01-01',
  dateTo: '2024-12-31'
});
```

#### Real-time Broadcasting
```javascript
// Broadcast custom alerts
await climateDbService.broadcastAnomalyAlert({
  id: 'alert-123',
  level: 'critical',
  message: 'Emergency weather event detected'
});
```

## 🔄 Data Initialization

### Automatic Population
Run the database initializer to populate with sample data:

```javascript
import dbInitializer from './utils/databaseInitializer';

// Initialize with sample data
await dbInitializer.initialize();
```

This will create:
- 90 days of historical data for 3 stations
- 3 sample climate anomalies
- Model predictions for 5 stations
- 2 sample alerts

### Manual Data Entry
You can also manually insert data through the Supabase Table Editor or use the service methods.

## 📈 Monitoring and Analytics

### Built-in Views
The schema includes helpful views:
- `climate_anomaly_stats`: Aggregated anomaly statistics
- `recent_predictions`: Recent model predictions with station info

### Performance Monitoring
- Query performance tracking
- Index usage statistics
- Connection pooling metrics

## 🔐 Security Best Practices

### Environment Variables
Never commit sensitive keys to version control:
```env
# .env (keep this file private)
DATABASE_URL=postgresql://postgres:nopassword@db.awxqkescyeetesyhdtxj.supabase.co:5432/postgres
REACT_APP_SUPABASE_URL=https://awxqkescyeetesyhdtxj.supabase.co
REACT_APP_SUPABASE_ANON_KEY=your_actual_key_here
```

### Row Level Security
Current policies allow:
- **Public read access** to weather and climate data
- **Authenticated write access** for data insertion
- **User-specific access** to sessions and logs

Modify these policies in Supabase dashboard under **Authentication** → **Policies**.

## 🐛 Troubleshooting

### Common Issues

#### Connection Problems
```javascript
// Test database connection
const isConnected = await dbInitializer.testConnection();
if (!isConnected) {
  console.error('Database connection failed');
}
```

#### Missing Tables
If tables don't exist, re-run the `schema.sql` file in Supabase SQL Editor.

#### Permission Errors
Check your RLS policies in Supabase dashboard under **Authentication** → **Policies**.

#### Real-time Not Working
Ensure your Supabase anon key has real-time permissions enabled.

## 📚 Advanced Usage

### Custom Queries
For complex queries not covered by the service:

```javascript
import { supabase } from './config/supabase';

const { data, error } = await supabase
  .from('climate_anomalies')
  .select(`
    *,
    weather_stations(name, country),
    anomaly_alerts(alert_level, status)
  `)
  .gte('confidence_score', 0.9)
  .order('detected_at', { ascending: false });
```

### Performance Optimization
- Use `select()` to limit returned columns
- Implement pagination for large datasets
- Use appropriate indexes for your query patterns
- Consider materialized views for complex aggregations

## 🔄 Migration and Backup

### Backup Strategy
- Supabase provides automatic backups
- Export critical data regularly
- Test restore procedures

### Schema Migrations
For schema changes:
1. Create migration SQL files
2. Test in development environment
3. Apply to production during maintenance windows

## 📞 Support

For issues with this database setup:
1. Check Supabase documentation: [https://supabase.com/docs](https://supabase.com/docs)
2. Review error logs in Supabase dashboard
3. Use the built-in SQL Editor for debugging queries

---

**Next Steps**: Once your database is set up, you can proceed with authentication setup and real-time features integration!