import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { FaHistory } from 'react-icons/fa';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter
} from 'recharts';

const HistoricalContainer = styled.div`
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 3rem;
  
  h1 {
    font-size: 2.5rem;
    color: white;
    margin-bottom: 1rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
  }
  
  p {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.9);
  }
`;

const FilterPanel = styled(motion.div)`
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 1.5rem;
  margin-bottom: 2rem;
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  align-items: center;
`;

const FilterGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  
  label {
    font-size: 0.9rem;
    font-weight: 600;
    color: #333;
  }
  
  select, input {
    padding: 0.5rem;
    border: 1px solid #ddd;
    border-radius: 6px;
    font-size: 0.9rem;
  }
`;

const DataCard = styled(motion.div)`
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 2rem;
  margin-bottom: 2rem;
  box-shadow: 0 8px 32px rgba(0,0,0,0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.2);
`;

const EventsList = styled.div`
  max-height: 400px;
  overflow-y: auto;
`;

const EventItem = styled.div`
  display: flex;
  align-items: center;
  padding: 1rem;
  margin-bottom: 0.5rem;
  border-radius: 8px;
  background: rgba(102, 126, 234, 0.05);
  border-left: 4px solid ${props => {
    switch(props.$severity) {
      case 'extreme': return '#F44336';
      case 'severe': return '#FF9800';
      case 'moderate': return '#2196F3';
      default: return '#4CAF50';
    }
  }};
  
  .event-date {
    min-width: 100px;
    font-weight: bold;
    color: #667eea;
  }
  
  .event-content {
    flex: 1;
    margin-left: 1rem;
    
    .event-title {
      font-weight: 600;
      margin-bottom: 0.25rem;
    }
    
    .event-description {
      font-size: 0.9rem;
      color: #666;
    }
  }
  
  .event-severity {
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: bold;
    text-transform: uppercase;
    background: ${props => {
      switch(props.$severity) {
        case 'extreme': return '#F44336';
        case 'severe': return '#FF9800';
        case 'moderate': return '#2196F3';
        default: return '#4CAF50';
      }
    }};
    color: white;
  }
`;

// Sample historical data
const historicalTrends = [
  { year: '2020', temperature: 1.2, precipitation: -0.8, co2: 415 },
  { year: '2021', temperature: 1.4, precipitation: -1.2, co2: 418 },
  { year: '2022', temperature: 1.6, precipitation: 0.5, co2: 421 },
  { year: '2023', temperature: 1.8, precipitation: 1.2, co2: 424 },
  { year: '2024', temperature: 2.1, precipitation: -0.3, co2: 427 }
];

const extremeEvents = [
  {
    id: 1,
    date: '2024-08-15',
    title: 'European Heat Dome',
    description: 'Record-breaking temperatures across Central Europe, 8°C above normal',
    severity: 'extreme',
    location: 'Central Europe'
  },
  {
    id: 2,
    date: '2024-07-22',
    title: 'Pacific Typhoon Cluster',
    description: 'Unusual formation of 3 simultaneous typhoons in Western Pacific',
    severity: 'severe',
    location: 'Western Pacific'
  },
  {
    id: 3,
    date: '2024-06-10',
    title: 'Arctic Ice Melt Acceleration',
    description: 'Greenland ice sheet losing mass 40% faster than previous decade',
    severity: 'severe',
    location: 'Arctic'
  },
  {
    id: 4,
    date: '2024-05-03',
    title: 'Sahara Dust Storm',
    description: 'Massive dust storm affecting air quality across North Africa',
    severity: 'moderate',
    location: 'North Africa'
  },
  {
    id: 5,
    date: '2024-04-18',
    title: 'Australian Coral Bleaching',
    description: 'Great Barrier Reef experiencing widespread coral bleaching event',
    severity: 'extreme',
    location: 'Australia'
  }
];

const anomalyFrequency = [
  { month: 'Jan', count: 12, severity: 2.3 },
  { month: 'Feb', count: 15, severity: 2.8 },
  { month: 'Mar', count: 18, severity: 3.1 },
  { month: 'Apr', count: 22, severity: 3.5 },
  { month: 'May', count: 28, severity: 4.2 },
  { month: 'Jun', count: 35, severity: 4.8 },
  { month: 'Jul', count: 42, severity: 5.1 },
  { month: 'Aug', count: 38, severity: 4.9 },
  { month: 'Sep', count: 31, severity: 4.3 },
  { month: 'Oct', count: 25, severity: 3.7 },
  { month: 'Nov', count: 19, severity: 3.2 },
  { month: 'Dec', count: 16, severity: 2.9 }
];

function HistoricalData() {
  const [selectedYear, setSelectedYear] = useState('2024');
  const [selectedRegion, setSelectedRegion] = useState('global');
  const [searchTerm, setSearchTerm] = useState('');

  const filteredEvents = extremeEvents.filter(event => {
    const matchesSearch = event.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         event.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesYear = event.date.startsWith(selectedYear);
    return matchesSearch && (selectedYear === 'all' || matchesYear);
  });

  return (
    <HistoricalContainer>
      <Header>
        <h1>Historical Climate Data</h1>
        <p>Comprehensive analysis of past climate events and long-term trends</p>
      </Header>

      <FilterPanel
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <FilterGroup>
          <label>Year</label>
          <select value={selectedYear} onChange={(e) => setSelectedYear(e.target.value)}>
            <option value="all">All Years</option>
            <option value="2024">2024</option>
            <option value="2023">2023</option>
            <option value="2022">2022</option>
            <option value="2021">2021</option>
            <option value="2020">2020</option>
          </select>
        </FilterGroup>
        
        <FilterGroup>
          <label>Region</label>
          <select value={selectedRegion} onChange={(e) => setSelectedRegion(e.target.value)}>
            <option value="global">Global</option>
            <option value="north_america">North America</option>
            <option value="europe">Europe</option>
            <option value="asia">Asia</option>
            <option value="africa">Africa</option>
            <option value="oceania">Oceania</option>
          </select>
        </FilterGroup>
        
        <FilterGroup>
          <label>Search Events</label>
          <input
            type="text"
            placeholder="Search by keyword..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </FilterGroup>
      </FilterPanel>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
        <DataCard
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <h2 style={{ marginBottom: '1.5rem' }}>Long-term Climate Trends</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={historicalTrends}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis yAxisId="temp" orientation="left" />
              <YAxis yAxisId="co2" orientation="right" />
              <Tooltip />
              <Legend />
              <Line yAxisId="temp" type="monotone" dataKey="temperature" stroke="#FF6B6B" name="Temperature Anomaly (°C)" />
              <Line yAxisId="temp" type="monotone" dataKey="precipitation" stroke="#4ECDC4" name="Precipitation Anomaly (%)" />
              <Line yAxisId="co2" type="monotone" dataKey="co2" stroke="#45B7D1" name="CO2 Levels (ppm)" />
            </LineChart>
          </ResponsiveContainer>
        </DataCard>

        <DataCard
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <h2 style={{ marginBottom: '1.5rem' }}>Monthly Anomaly Patterns</h2>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={anomalyFrequency}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis yAxisId="count" orientation="left" />
              <YAxis yAxisId="severity" orientation="right" />
              <Tooltip />
              <Legend />
              <Scatter yAxisId="count" dataKey="count" fill="#667eea" name="Anomaly Count" />
              <Scatter yAxisId="severity" dataKey="severity" fill="#F44336" name="Average Severity" />
            </ScatterChart>
          </ResponsiveContainer>
        </DataCard>
      </div>

      <DataCard
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <h2 style={{ marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FaHistory style={{ color: '#667eea' }} />
          Extreme Weather Events ({filteredEvents.length} events found)
        </h2>
        
        <EventsList>
          {filteredEvents.map((event) => (
            <EventItem key={event.id} $severity={event.severity}>
              <div className="event-date">{event.date}</div>
              <div className="event-content">
                <div className="event-title">{event.title}</div>
                <div className="event-description">{event.description}</div>
                <div style={{ fontSize: '0.8rem', color: '#999', marginTop: '0.25rem' }}>
                  Location: {event.location}
                </div>
              </div>
              <div className="event-severity">{event.severity}</div>
            </EventItem>
          ))}
        </EventsList>
      </DataCard>
    </HistoricalContainer>
  );
}

export default HistoricalData;