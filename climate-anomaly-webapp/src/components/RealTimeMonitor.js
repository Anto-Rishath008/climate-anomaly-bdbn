import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { FaPlay, FaPause, FaSyncAlt, FaDownload } from 'react-icons/fa';
import WeatherCard from './WeatherCard';

const MonitorContainer = styled.div`
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

const ControlPanel = styled(motion.div)`
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 1.5rem;
  margin-bottom: 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
`;

const ControlButton = styled.button`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  background: ${props => props.primary ? '#667eea' : '#f8f9fa'};
  color: ${props => props.primary ? 'white' : '#333'};
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  }
`;

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  .status-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: ${props => props.active ? '#4CAF50' : '#999'};
    animation: ${props => props.active ? 'pulse 2s infinite' : 'none'};
  }
  
  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }
`;

const WeatherGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
`;

function RealTimeMonitor() {
  const [isMonitoring, setIsMonitoring] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [weatherData, setWeatherData] = useState([
    {
      id: 1,
      location: 'New York, USA',
      temperature: 24,
      humidity: 68,
      pressure: 1013,
      windSpeed: 12,
      anomalyScore: 0.23,
      status: 'normal'
    },
    {
      id: 2,
      location: 'London, UK',
      temperature: 18,
      humidity: 82,
      pressure: 998,
      windSpeed: 8,
      anomalyScore: 0.67,
      status: 'warning'
    },
    {
      id: 3,
      location: 'Tokyo, Japan',
      temperature: 28,
      humidity: 75,
      pressure: 1008,
      windSpeed: 15,
      anomalyScore: 0.15,
      status: 'normal'
    },
    {
      id: 4,
      location: 'Sydney, Australia',
      temperature: 22,
      humidity: 45,
      pressure: 1020,
      windSpeed: 20,
      anomalyScore: 0.89,
      status: 'critical'
    }
  ]);

  useEffect(() => {
    let interval;
    if (isMonitoring) {
      interval = setInterval(() => {
        // Simulate real-time data updates
        setWeatherData(prev => prev.map(item => ({
          ...item,
          temperature: item.temperature + (Math.random() - 0.5) * 2,
          humidity: Math.max(0, Math.min(100, item.humidity + (Math.random() - 0.5) * 5)),
          pressure: item.pressure + (Math.random() - 0.5) * 10,
          windSpeed: Math.max(0, item.windSpeed + (Math.random() - 0.5) * 3),
          anomalyScore: Math.max(0, Math.min(1, item.anomalyScore + (Math.random() - 0.5) * 0.1))
        })));
        setLastUpdate(new Date());
      }, 5000); // Update every 5 seconds
    }
    return () => clearInterval(interval);
  }, [isMonitoring]);

  const toggleMonitoring = () => {
    setIsMonitoring(!isMonitoring);
  };

  const refreshData = () => {
    setLastUpdate(new Date());
    // Simulate data refresh
  };

  const exportData = () => {
    const dataStr = JSON.stringify(weatherData, null, 2);
    const dataBlob = new Blob([dataStr], {type:'application/json'});
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `weather-data-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
  };

  return (
    <MonitorContainer>
      <Header>
        <h1>Real-Time Climate Monitor</h1>
        <p>Live monitoring of global weather conditions and anomaly detection</p>
      </Header>

      <ControlPanel
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <StatusIndicator active={isMonitoring}>
          <div className="status-dot"></div>
          <span>
            {isMonitoring ? 'Monitoring Active' : 'Monitoring Paused'} • 
            Last update: {lastUpdate.toLocaleTimeString()}
          </span>
        </StatusIndicator>
        
        <div style={{ display: 'flex', gap: '1rem' }}>
          <ControlButton primary onClick={toggleMonitoring}>
            {isMonitoring ? <FaPause /> : <FaPlay />}
            {isMonitoring ? 'Pause' : 'Start'} Monitoring
          </ControlButton>
          
          <ControlButton onClick={refreshData}>
            <FaSyncAlt />
            Refresh
          </ControlButton>
          
          <ControlButton onClick={exportData}>
            <FaDownload />
            Export Data
          </ControlButton>
        </div>
      </ControlPanel>

      <WeatherGrid>
        {weatherData.map((data, index) => (
          <motion.div
            key={data.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: index * 0.1 }}
          >
            <WeatherCard data={data} />
          </motion.div>
        ))}
      </WeatherGrid>
    </MonitorContainer>
  );
}

export default RealTimeMonitor;