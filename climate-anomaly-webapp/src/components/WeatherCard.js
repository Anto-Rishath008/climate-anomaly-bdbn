import React from 'react';
import styled from 'styled-components';
import { FaThermometerHalf, FaTint, FaWind, FaCompress, FaExclamationTriangle } from 'react-icons/fa';

const Card = styled.div`
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 1.5rem;
  box-shadow: 0 8px 32px rgba(0,0,0,0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.2);
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.15);
  }
`;

const CardHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  
  h3 {
    margin: 0;
    color: #333;
    font-size: 1.2rem;
  }
`;

const StatusBadge = styled.div`
  padding: 0.25rem 0.75rem;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: bold;
  text-transform: uppercase;
  background: ${props => {
    switch(props.status) {
      case 'critical': return '#F44336';
      case 'warning': return '#FF9800';
      default: return '#4CAF50';
    }
  }};
  color: white;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1.5rem;
`;

const Metric = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  svg {
    color: #667eea;
    font-size: 1.1rem;
  }
  
  .metric-info {
    flex: 1;
    
    .metric-label {
      font-size: 0.8rem;
      color: #666;
      margin-bottom: 0.25rem;
    }
    
    .metric-value {
      font-size: 1.1rem;
      font-weight: bold;
      color: #333;
    }
  }
`;

const AnomalyScore = styled.div`
  padding: 1rem;
  border-radius: 8px;
  background: ${props => {
    if (props.score > 0.7) return 'rgba(244, 67, 54, 0.1)';
    if (props.score > 0.4) return 'rgba(255, 152, 0, 0.1)';
    return 'rgba(76, 175, 80, 0.1)';
  }};
  border-left: 4px solid ${props => {
    if (props.score > 0.7) return '#F44336';
    if (props.score > 0.4) return '#FF9800';
    return '#4CAF50';
  }};
  
  .score-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
    
    svg {
      color: ${props => {
        if (props.score > 0.7) return '#F44336';
        if (props.score > 0.4) return '#FF9800';
        return '#4CAF50';
      }};
    }
    
    .score-label {
      font-size: 0.9rem;
      font-weight: 600;
      color: #333;
    }
  }
  
  .score-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: ${props => {
      if (props.score > 0.7) return '#F44336';
      if (props.score > 0.4) return '#FF9800';
      return '#4CAF50';
    }};
  }
  
  .score-description {
    font-size: 0.8rem;
    color: #666;
    margin-top: 0.25rem;
  }
`;

function WeatherCard({ data }) {
  const getAnomalyDescription = (score) => {
    if (score > 0.7) return 'High anomaly detected - immediate attention required';
    if (score > 0.4) return 'Moderate anomaly - monitoring recommended';
    return 'Normal conditions - no anomalies detected';
  };

  const getStatusFromScore = (score) => {
    if (score > 0.7) return 'critical';
    if (score > 0.4) return 'warning';
    return 'normal';
  };

  return (
    <Card>
      <CardHeader>
        <h3>{data.location}</h3>
        <StatusBadge status={getStatusFromScore(data.anomalyScore)}>
          {getStatusFromScore(data.anomalyScore)}
        </StatusBadge>
      </CardHeader>
      
      <MetricsGrid>
        <Metric>
          <FaThermometerHalf />
          <div className="metric-info">
            <div className="metric-label">Temperature</div>
            <div className="metric-value">{data.temperature.toFixed(1)}°C</div>
          </div>
        </Metric>
        
        <Metric>
          <FaTint />
          <div className="metric-info">
            <div className="metric-label">Humidity</div>
            <div className="metric-value">{data.humidity.toFixed(0)}%</div>
          </div>
        </Metric>
        
        <Metric>
          <FaCompress />
          <div className="metric-info">
            <div className="metric-label">Pressure</div>
            <div className="metric-value">{data.pressure.toFixed(0)} hPa</div>
          </div>
        </Metric>
        
        <Metric>
          <FaWind />
          <div className="metric-info">
            <div className="metric-label">Wind Speed</div>
            <div className="metric-value">{data.windSpeed.toFixed(1)} km/h</div>
          </div>
        </Metric>
      </MetricsGrid>
      
      <AnomalyScore score={data.anomalyScore}>
        <div className="score-header">
          <FaExclamationTriangle />
          <span className="score-label">Anomaly Score</span>
        </div>
        <div className="score-value">{(data.anomalyScore * 100).toFixed(1)}%</div>
        <div className="score-description">
          {getAnomalyDescription(data.anomalyScore)}
        </div>
      </AnomalyScore>
    </Card>
  );
}

export default WeatherCard;