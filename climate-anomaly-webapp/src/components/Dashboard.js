import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  FaThermometerHalf, 
  FaCloudRain, 
  FaWind, 
  FaExclamationTriangle,
  FaCheckCircle,
  FaTimesCircle,
  FaSearch,
  FaMapMarkerAlt,
  FaClock,
  FaEye
} from 'react-icons/fa';
import { theme, threatLevels, weatherIcons } from '../theme';
import WorldMap from './WorldMap';
import AnomalyChart from './AnomalyChart';

const DashboardContainer = styled.div`
  padding: 2rem;
  max-width: 1600px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 2rem;
  min-height: calc(100vh - 160px);
  
  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }
`;

const MapSection = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
`;

const SearchBar = styled.div`
  position: relative;
  
  input {
    width: 100%;
    padding: 1rem 1rem 1rem 3rem;
    background: ${theme.glassmorphism.light.background};
    backdrop-filter: ${theme.glassmorphism.light.backdropFilter};
    border: ${theme.glassmorphism.light.border};
    border-radius: ${theme.borderRadius.lg};
    color: ${theme.colors.text.primary};
    font-size: 1.1rem;
    transition: all ${theme.animation.duration.normal} ease;
    
    &::placeholder {
      color: ${theme.colors.text.muted};
    }
    
    &:focus {
      outline: none;
      border-color: ${theme.colors.accent.info};
      background: ${theme.glassmorphism.medium.background};
      box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.1);
    }
  }
  
  .search-icon {
    position: absolute;
    left: 1rem;
    top: 50%;
    transform: translateY(-50%);
    color: ${theme.colors.text.muted};
    font-size: 1.2rem;
  }
`;

const MapContainer = styled(motion.div)`
  background: ${theme.glassmorphism.light.background};
  backdrop-filter: ${theme.glassmorphism.light.backdropFilter};
  border: ${theme.glassmorphism.light.border};
  border-radius: ${theme.borderRadius.xl};
  padding: 1.5rem;
  flex: 1;
  min-height: 500px;
  position: relative;
  overflow: hidden;
  
  .map-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: ${theme.colors.text.secondary};
    font-size: 1.2rem;
    text-align: center;
    
    .globe-icon {
      font-size: 4rem;
      margin-bottom: 1rem;
      color: ${theme.colors.accent.info};
    }
  }
`;

const RightPanel = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
`;

const WeatherCard = styled(motion.div)`
  background: ${theme.glassmorphism.medium.background};
  backdrop-filter: ${theme.glassmorphism.medium.backdropFilter};
  border: ${theme.glassmorphism.medium.border};
  border-radius: ${theme.borderRadius.xl};
  padding: 2rem;
  
  .location {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: ${theme.colors.text.secondary};
    margin-bottom: 1.5rem;
    
    .location-icon {
      color: ${theme.colors.accent.info};
    }
  }
  
  .current-weather {
    text-align: center;
    margin-bottom: 2rem;
    
    .weather-icon {
      font-size: 4rem;
      margin-bottom: 1rem;
    }
    
    .temperature {
      font-size: 3rem;
      font-weight: ${theme.typography.fontWeight.light};
      color: ${theme.colors.text.primary};
      margin-bottom: 0.5rem;
    }
    
    .condition {
      color: ${theme.colors.text.secondary};
      font-size: 1.1rem;
    }
  }
  
  .weather-details {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    
    .detail-item {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: ${theme.colors.text.secondary};
      
      .detail-icon {
        color: ${theme.colors.accent.info};
      }
      
      .detail-value {
        color: ${theme.colors.text.primary};
        font-weight: ${theme.typography.fontWeight.medium};
      }
    }
  }
`;

const PredictionPanel = styled(motion.div)`
  background: ${theme.glassmorphism.light.background};
  backdrop-filter: ${theme.glassmorphism.light.backdropFilter};
  border: ${theme.glassmorphism.light.border};
  border-radius: ${theme.borderRadius.xl};
  padding: 2rem;
  
  h3 {
    color: ${theme.colors.text.primary};
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    
    .prediction-icon {
      color: ${theme.colors.accent.info};
    }
  }
  
  .prediction-form {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 1.5rem;
    
    input {
      padding: 0.8rem 1rem;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: ${theme.borderRadius.md};
      color: ${theme.colors.text.primary};
      
      &::placeholder {
        color: ${theme.colors.text.muted};
      }
      
      &:focus {
        outline: none;
        border-color: ${theme.colors.accent.info};
        background: rgba(255, 255, 255, 0.15);
      }
    }
    
    button {
      padding: 0.8rem 1.5rem;
      background: linear-gradient(135deg, ${theme.colors.accent.info} 0%, ${theme.colors.primary[600]} 100%);
      border: none;
      border-radius: ${theme.borderRadius.md};
      color: white;
      font-weight: ${theme.typography.fontWeight.medium};
      cursor: pointer;
      transition: all ${theme.animation.duration.normal} ease;
      
      &:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(96, 165, 250, 0.3);
      }
    }
  }
`;

const AnomalyReport = styled(motion.div)`
  background: ${theme.glassmorphism.medium.background};
  backdrop-filter: ${theme.glassmorphism.medium.backdropFilter};
  border: ${theme.glassmorphism.medium.border};
  border-radius: ${theme.borderRadius.xl};
  padding: 2rem;
  
  .report-header {
    display: flex;
    align-items: center;
    justify-content: between;
    margin-bottom: 1.5rem;
    
    h3 {
      color: ${theme.colors.text.primary};
      margin: 0;
    }
    
    .threat-badge {
      padding: 0.5rem 1rem;
      border-radius: 20px;
      font-size: 0.9rem;
      font-weight: ${theme.typography.fontWeight.medium};
      margin-left: auto;
    }
  }
  
  .report-content {
    color: ${theme.colors.text.secondary};
    line-height: 1.6;
    
    .prediction-summary {
      font-size: 1.1rem;
      margin-bottom: 1rem;
      color: ${theme.colors.text.primary};
    }
    
    .details {
      font-size: 0.95rem;
    }
  }
`;

const SectionCard = styled(motion.div)`
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 8px 32px rgba(0,0,0,0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.2);
  
  h2 {
    margin-bottom: 1.5rem;
    color: #333;
    display: flex;
    align-items: center;
    
    svg {
      margin-right: 0.75rem;
      color: #667eea;
    }
  }
`;

const AlertsList = styled.div`
  max-height: 300px;
  overflow-y: auto;
`;

const AlertItem = styled.div`
  display: flex;
  align-items: center;
  padding: 1rem;
  margin-bottom: 0.5rem;
  border-radius: 8px;
  background: ${props => {
    switch(props.$severity) {
      case 'critical': return 'rgba(244, 67, 54, 0.1)';
      case 'warning': return 'rgba(255, 152, 0, 0.1)';
      default: return 'rgba(76, 175, 80, 0.1)';
    }
  }};
  border-left: 4px solid ${props => {
    switch(props.$severity) {
      case 'critical': return '#F44336';
      case 'warning': return '#FF9800';
      default: return '#4CAF50';
    }
  }};
  
  svg {
    margin-right: 0.75rem;
    color: ${props => {
      switch(props.$severity) {
        case 'critical': return '#F44336';
        case 'warning': return '#FF9800';
        default: return '#4CAF50';
      }
    }};
  }
  
  .alert-content {
    flex: 1;
    
    .alert-title {
      font-weight: 600;
      margin-bottom: 0.25rem;
    }
    
    .alert-details {
      font-size: 0.9rem;
      color: #666;
    }
  }
  
  .alert-time {
    font-size: 0.8rem;
    color: #999;
  }
`;

function Dashboard() {
  const [stats, setStats] = useState({
    anomaliesDetected: 12,
    systemAccuracy: 97.8,
    dataPoints: 156420,
    lastUpdate: new Date()
  });

  const [alerts] = useState([
    {
      id: 1,
      severity: 'warning',
      title: 'Temperature Anomaly Detected',
      details: 'Unusual temperature spike in Northern Europe',
      time: '5 min ago'
    },
    {
      id: 2,
      severity: 'normal',
      title: 'System Health Check',
      details: 'All monitoring systems operational',
      time: '15 min ago'
    },
    {
      id: 3,
      severity: 'critical',
      title: 'Precipitation Pattern Alert',
      details: 'Extreme rainfall detected in Southeast Asia',
      time: '1 hour ago'
    }
  ]);

  useEffect(() => {
    // Simulate real-time data updates
    const interval = setInterval(() => {
      setStats(prev => ({
        ...prev,
        anomaliesDetected: prev.anomaliesDetected + Math.floor(Math.random() * 3),
        lastUpdate: new Date()
      }));
    }, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const getSeverityIcon = (severity) => {
    switch(severity) {
      case 'critical': return <FaTimesCircle />;
      case 'warning': return <FaExclamationTriangle />;
      default: return <FaCheckCircle />;
    }
  };

  return (
    <DashboardContainer>
      <WelcomeSection
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1>Climate Anomaly Detection System</h1>
        <p>
          Real-time monitoring and prediction of climate anomalies using 
          Bayesian Deep Learning with 97.8% accuracy
        </p>
      </WelcomeSection>

      <StatsGrid>
        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          whileHover={{ y: -5 }}
        >
          <div className="stat-header">
            <FaExclamationTriangle />
            <h3>Anomalies Detected</h3>
          </div>
          <div className="stat-value">{stats.anomaliesDetected}</div>
          <div className="stat-change positive">+3 in last hour</div>
        </StatCard>

        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          whileHover={{ y: -5 }}
        >
          <div className="stat-header">
            <FaCheckCircle />
            <h3>System Accuracy</h3>
          </div>
          <div className="stat-value">{stats.systemAccuracy}%</div>
          <div className="stat-change positive">+0.2% from last week</div>
        </StatCard>

        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          whileHover={{ y: -5 }}
        >
          <div className="stat-header">
            <FaThermometerHalf />
            <h3>Data Points</h3>
          </div>
          <div className="stat-value">{stats.dataPoints.toLocaleString()}</div>
          <div className="stat-change neutral">Processed today</div>
        </StatCard>

        <StatCard
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          whileHover={{ y: -5 }}
        >
          <div className="stat-header">
            <FaWind />
            <h3>Active Monitors</h3>
          </div>
          <div className="stat-value">24</div>
          <div className="stat-change positive">All systems online</div>
        </StatCard>
      </StatsGrid>

      <MainContent>
        <SectionCard
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
        >
          <h2>
            <FaExclamationTriangle />
            Recent Alerts
          </h2>
          <AlertsList>
            {alerts.map((alert) => (
              <AlertItem key={alert.id} $severity={alert.severity}>
                {getSeverityIcon(alert.severity)}
                <div className="alert-content">
                  <div className="alert-title">{alert.title}</div>
                  <div className="alert-details">{alert.details}</div>
                </div>
                <div className="alert-time">{alert.time}</div>
              </AlertItem>
            ))}
          </AlertsList>
        </SectionCard>

        <SectionCard
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          <h2>
            <FaCloudRain />
            Anomaly Trends
          </h2>
          <AnomalyChart />
        </SectionCard>
      </MainContent>

      <SectionCard
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.7 }}
      >
        <h2>Global Climate Monitoring</h2>
        <WorldMap />
      </SectionCard>
    </DashboardContainer>
  );
}

export default Dashboard;