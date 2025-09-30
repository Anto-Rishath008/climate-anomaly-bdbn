import React from 'react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';

// Sample data for the last 24 hours
const anomalyData = [
  { time: '00:00', temperature: 2, precipitation: 1, pressure: 0, humidity: 1 },
  { time: '02:00', temperature: 3, precipitation: 2, pressure: 1, humidity: 2 },
  { time: '04:00', temperature: 1, precipitation: 4, pressure: 2, humidity: 1 },
  { time: '06:00', temperature: 5, precipitation: 3, pressure: 1, humidity: 3 },
  { time: '08:00', temperature: 7, precipitation: 2, pressure: 3, humidity: 2 },
  { time: '10:00', temperature: 4, precipitation: 1, pressure: 2, humidity: 4 },
  { time: '12:00', temperature: 6, precipitation: 5, pressure: 4, humidity: 3 },
  { time: '14:00', temperature: 8, precipitation: 7, pressure: 3, humidity: 5 },
  { time: '16:00', temperature: 9, precipitation: 6, pressure: 5, humidity: 4 },
  { time: '18:00', temperature: 7, precipitation: 4, pressure: 6, humidity: 6 },
  { time: '20:00', temperature: 5, precipitation: 3, pressure: 4, humidity: 3 },
  { time: '22:00', temperature: 3, precipitation: 2, pressure: 2, humidity: 2 }
];

// Custom tooltip
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: 'rgba(255, 255, 255, 0.95)',
        border: '1px solid #ccc',
        borderRadius: '8px',
        padding: '12px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
      }}>
        <p style={{ margin: '0 0 8px 0', fontWeight: 'bold' }}>{`Time: ${label}`}</p>
        {payload.map((entry, index) => (
          <p key={index} style={{ 
            margin: '4px 0', 
            color: entry.color,
            textTransform: 'capitalize'
          }}>
            {`${entry.dataKey}: ${entry.value} anomalies`}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

function AnomalyChart() {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <AreaChart
        data={anomalyData}
        margin={{
          top: 10,
          right: 30,
          left: 0,
          bottom: 0,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
        <XAxis 
          dataKey="time" 
          stroke="#666"
          fontSize={12}
        />
        <YAxis 
          stroke="#666"
          fontSize={12}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend />
        
        <Area
          type="monotone"
          dataKey="temperature"
          stackId="1"
          stroke="#FF6B6B"
          fill="rgba(255, 107, 107, 0.3)"
          name="Temperature"
        />
        <Area
          type="monotone"
          dataKey="precipitation"
          stackId="1"
          stroke="#4ECDC4"
          fill="rgba(78, 205, 196, 0.3)"
          name="Precipitation"
        />
        <Area
          type="monotone"
          dataKey="pressure"
          stackId="1"
          stroke="#45B7D1"
          fill="rgba(69, 183, 209, 0.3)"
          name="Pressure"
        />
        <Area
          type="monotone"
          dataKey="humidity"
          stackId="1"
          stroke="#F9CA24"
          fill="rgba(249, 202, 36, 0.3)"
          name="Humidity"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export default AnomalyChart;