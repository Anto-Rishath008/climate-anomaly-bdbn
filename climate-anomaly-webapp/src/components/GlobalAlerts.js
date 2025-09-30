import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  FaExclamationTriangle, 
  FaFilter,
  FaClock,
  FaMapMarkerAlt,
  FaSearch,
  FaGlobe,
  FaCloud,
  FaBolt,
  FaSnowflake,
  FaSun,
  FaWater
} from 'react-icons/fa';
import { theme, threatLevels } from '../theme';

const AlertsContainer = styled.div`
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
`;

const AlertsHeader = styled.div`
  margin-bottom: 2rem;
  
  h1 {
    color: ${theme.colors.text.primary};
    font-size: 2.5rem;
    font-weight: ${theme.typography.fontWeight.light};
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    
    .header-icon {
      color: ${theme.colors.accent.warning};
    }
  }
  
  p {
    color: ${theme.colors.text.secondary};
    font-size: 1.1rem;
  }
`;

const FilterBar = styled(motion.div)`
  background: ${theme.glassmorphism.light.background};
  backdrop-filter: ${theme.glassmorphism.light.backdropFilter};
  border: ${theme.glassmorphism.light.border};
  border-radius: ${theme.borderRadius.lg};
  padding: 1.5rem;
  margin-bottom: 2rem;
  
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  align-items: center;
  
  .filter-group {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    
    label {
      color: ${theme.colors.text.secondary};
      font-size: 0.9rem;
    }
    
    select {
      padding: 0.5rem 1rem;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: ${theme.borderRadius.md};
      color: ${theme.colors.text.primary};
      cursor: pointer;
      
      option {
        background: ${theme.colors.secondary[800]};
        color: ${theme.colors.text.primary};
      }
      
      &:focus {
        outline: none;
        border-color: ${theme.colors.accent.info};
      }
    }
  }
  
  .search-filter {
    flex: 1;
    min-width: 250px;
    position: relative;
    
    input {
      width: 100%;
      padding: 0.5rem 0.5rem 0.5rem 2.5rem;
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
    
    .search-icon {
      position: absolute;
      left: 0.8rem;
      top: 50%;
      transform: translateY(-50%);
      color: ${theme.colors.text.muted};
    }
  }
`;

const AlertsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 1.5rem;
`;

const AlertCard = styled(motion.div)`
  background: ${theme.glassmorphism.medium.background};
  backdrop-filter: ${theme.glassmorphism.medium.backdropFilter};
  border: ${theme.glassmorphism.medium.border};
  border-radius: ${theme.borderRadius.lg};
  padding: 1.5rem;
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: ${props => threatLevels[props.severity]?.color || theme.colors.accent.info};
  }
  
  .alert-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 1rem;
    
    .alert-icon {
      font-size: 1.5rem;
      color: ${props => threatLevels[props.severity]?.color || theme.colors.accent.info};
      margin-right: 1rem;
    }
    
    .alert-title {
      flex: 1;
      
      h3 {
        color: ${theme.colors.text.primary};
        font-size: 1.1rem;
        font-weight: ${theme.typography.fontWeight.semibold};
        margin: 0 0 0.5rem 0;
      }
      
      .event-type {
        color: ${props => threatLevels[props.severity]?.color || theme.colors.accent.info};
        font-size: 0.85rem;
        font-weight: ${theme.typography.fontWeight.medium};
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
    }
    
    .severity-badge {
      padding: 0.3rem 0.8rem;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: ${theme.typography.fontWeight.medium};
      text-transform: uppercase;
      letter-spacing: 0.5px;
      background: ${props => threatLevels[props.severity]?.background || 'rgba(96, 165, 250, 0.1)'};
      color: ${props => threatLevels[props.severity]?.color || theme.colors.accent.info};
      border: 1px solid ${props => `${threatLevels[props.severity]?.color || theme.colors.accent.info}30`};
    }
  }
  
  .alert-details {
    margin-bottom: 1rem;
    
    .description {
      color: ${theme.colors.text.secondary};
      line-height: 1.6;
      margin-bottom: 1rem;
    }
    
    .location {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: ${theme.colors.text.muted};
      font-size: 0.9rem;
      margin-bottom: 0.5rem;
      
      .location-icon {
        color: ${theme.colors.accent.info};
      }
    }
    
    .timeframe {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: ${theme.colors.text.muted};
      font-size: 0.9rem;
      
      .time-icon {
        color: ${theme.colors.accent.info};
      }
    }
  }
  
  .alert-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 1rem;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    
    .alert-time {
      color: ${theme.colors.text.muted};
      font-size: 0.85rem;
    }
    
    .alert-actions {
      display: flex;
      gap: 0.5rem;
      
      button {
        padding: 0.4rem 0.8rem;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: ${theme.borderRadius.sm};
        color: ${theme.colors.text.secondary};
        font-size: 0.8rem;
        cursor: pointer;
        transition: all ${theme.animation.duration.normal} ease;
        
        &:hover {
          background: rgba(255, 255, 255, 0.15);
          color: ${theme.colors.text.primary};
        }
      }
    }
  }
`;

const StatsBar = styled(motion.div)`
  background: ${theme.glassmorphism.light.background};
  backdrop-filter: ${theme.glassmorphism.light.backdropFilter};
  border: ${theme.glassmorphism.light.border};
  border-radius: ${theme.borderRadius.lg};
  padding: 1.5rem;
  margin-bottom: 2rem;
  
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 2rem;
  
  .stat-item {
    text-align: center;
    
    .stat-value {
      font-size: 2rem;
      font-weight: ${theme.typography.fontWeight.bold};
      color: ${theme.colors.text.primary};
      margin-bottom: 0.5rem;
    }
    
    .stat-label {
      color: ${theme.colors.text.secondary};
      font-size: 0.9rem;
    }
  }
`;

function GlobalAlerts() {
  const [alerts, setAlerts] = useState([]);
  const [filteredAlerts, setFilteredAlerts] = useState([]);
  const [expandedDetails, setExpandedDetails] = useState(new Set());
  const [trackedAlerts, setTrackedAlerts] = useState(new Set());
  const [filters, setFilters] = useState({
    region: 'all',
    severity: 'all',
    type: 'all',
    search: ''
  });

  const eventTypes = {
    hurricane: { icon: FaCloud, label: 'Hurricane' },
    tornado: { icon: FaBolt, label: 'Tornado' },
    flood: { icon: FaWater, label: 'Flood' },
    heatwave: { icon: FaSun, label: 'Heat Wave' },
    blizzard: { icon: FaSnowflake, label: 'Blizzard' },
    thunderstorm: { icon: FaBolt, label: 'Thunderstorm' }
  };

  useEffect(() => {
    // Generate sample alerts
    const sampleAlerts = [
      {
        id: 1,
        type: 'hurricane',
        title: 'Category 4 Hurricane Approaching',
        description: 'Major hurricane with sustained winds of 150 mph expected to make landfall within 48 hours. Potential for catastrophic damage and life-threatening storm surge.',
        region: 'North America',
        location: 'Gulf Coast, United States',
        timeframe: 'Next 48 hours',
        severity: 'severe',
        timestamp: '15 minutes ago'
      },
      {
        id: 2,
        type: 'heatwave',
        title: 'Extreme Heat Warning',
        description: 'Unprecedented temperature spike expected to reach 45°C (113°F) for 5 consecutive days. High risk of heat-related illnesses.',
        region: 'Europe',
        location: 'Southern Spain, Portugal',
        timeframe: 'July 15-20, 2025',
        severity: 'high',
        timestamp: '1 hour ago'
      },
      {
        id: 3,
        type: 'flood',
        title: 'Flash Flood Emergency',
        description: 'Rapid river rise due to intense rainfall. Immediate evacuation recommended for low-lying areas.',
        region: 'Asia',
        location: 'Northern India',
        timeframe: 'Currently active',
        severity: 'extreme',
        timestamp: '30 minutes ago'
      },
      {
        id: 4,
        type: 'tornado',
        title: 'Tornado Watch Issued',
        description: 'Conditions favorable for tornado development. Multiple supercells identified on radar.',
        region: 'North America',
        location: 'Midwest United States',
        timeframe: 'Next 6 hours',
        severity: 'moderate',
        timestamp: '2 hours ago'
      },
      {
        id: 5,
        type: 'blizzard',
        title: 'Severe Blizzard Conditions',
        description: 'Heavy snowfall with winds up to 70 mph creating whiteout conditions. Travel impossible.',
        region: 'North America',
        location: 'Canadian Prairies',
        timeframe: 'Next 24 hours',
        severity: 'high',
        timestamp: '45 minutes ago'
      },
      {
        id: 6,
        type: 'thunderstorm',
        title: 'Severe Thunderstorm Complex',
        description: 'Large thunderstorm system with damaging winds, large hail, and potential for isolated tornadoes.',
        region: 'South America',
        location: 'Northern Argentina',
        timeframe: 'Next 12 hours',
        severity: 'moderate',
        timestamp: '3 hours ago'
      }
    ];
    
    setAlerts(sampleAlerts);
    setFilteredAlerts(sampleAlerts);
  }, []);

  useEffect(() => {
    let filtered = alerts;

    // Apply filters
    if (filters.region !== 'all') {
      filtered = filtered.filter(alert => alert.region === filters.region);
    }

    if (filters.severity !== 'all') {
      filtered = filtered.filter(alert => alert.severity === filters.severity);
    }

    if (filters.type !== 'all') {
      filtered = filtered.filter(alert => alert.type === filters.type);
    }

    if (filters.search) {
      filtered = filtered.filter(alert => 
        alert.title.toLowerCase().includes(filters.search.toLowerCase()) ||
        alert.description.toLowerCase().includes(filters.search.toLowerCase()) ||
        alert.location.toLowerCase().includes(filters.search.toLowerCase())
      );
    }

    setFilteredAlerts(filtered);
  }, [filters, alerts]);

  const handleFilterChange = (filterType, value) => {
    setFilters(prev => ({
      ...prev,
      [filterType]: value
    }));
  };

  const getAlertStats = () => {
    const total = alerts.length;
    const severe = alerts.filter(alert => alert.severity === 'severe' || alert.severity === 'extreme').length;
    const active = alerts.filter(alert => alert.timeframe.includes('Currently') || alert.timeframe.includes('Next')).length;
    const regions = [...new Set(alerts.map(alert => alert.region))].length;

    return { total, severe, active, regions };
  };

  const stats = getAlertStats();

  const getEventIcon = (type) => {
    const EventIcon = eventTypes[type]?.icon || FaExclamationTriangle;
    return <EventIcon />;
  };

  return (
    <AlertsContainer>
      <AlertsHeader>
        <h1>
          <FaExclamationTriangle className="header-icon" />
          Global Alerts Feed
        </h1>
        <p>Real-time monitoring of predicted weather events and natural disasters worldwide</p>
      </AlertsHeader>

      <StatsBar
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="stat-item">
          <div className="stat-value">{stats.total}</div>
          <div className="stat-label">Total Alerts</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">{stats.severe}</div>
          <div className="stat-label">High Severity</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">{stats.active}</div>
          <div className="stat-label">Active Now</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">{stats.regions}</div>
          <div className="stat-label">Regions Affected</div>
        </div>
      </StatsBar>

      <FilterBar
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <div className="filter-group">
          <FaGlobe />
          <label>Region:</label>
          <select 
            value={filters.region} 
            onChange={(e) => handleFilterChange('region', e.target.value)}
          >
            <option value="all">All Regions</option>
            <option value="North America">North America</option>
            <option value="Europe">Europe</option>
            <option value="Asia">Asia</option>
            <option value="South America">South America</option>
            <option value="Africa">Africa</option>
            <option value="Oceania">Oceania</option>
          </select>
        </div>

        <div className="filter-group">
          <FaFilter />
          <label>Severity:</label>
          <select 
            value={filters.severity} 
            onChange={(e) => handleFilterChange('severity', e.target.value)}
          >
            <option value="all">All Levels</option>
            <option value="low">Low Risk</option>
            <option value="moderate">Moderate Risk</option>
            <option value="high">High Risk</option>
            <option value="severe">Severe Danger</option>
            <option value="extreme">Extreme Danger</option>
          </select>
        </div>

        <div className="filter-group">
          <FaCloud />
          <label>Type:</label>
          <select 
            value={filters.type} 
            onChange={(e) => handleFilterChange('type', e.target.value)}
          >
            <option value="all">All Types</option>
            <option value="hurricane">Hurricane</option>
            <option value="tornado">Tornado</option>
            <option value="flood">Flood</option>
            <option value="heatwave">Heat Wave</option>
            <option value="blizzard">Blizzard</option>
            <option value="thunderstorm">Thunderstorm</option>
          </select>
        </div>

        <div className="search-filter">
          <FaSearch className="search-icon" />
          <input
            type="text"
            placeholder="Search alerts..."
            value={filters.search}
            onChange={(e) => handleFilterChange('search', e.target.value)}
          />
        </div>
      </FilterBar>

      <AlertsGrid>
        {filteredAlerts.map((alert, index) => (
          <AlertCard
            key={alert.id}
            severity={alert.severity}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: index * 0.1 }}
            whileHover={{ y: -5, transition: { duration: 0.2 } }}
          >
            <div className="alert-header">
              <div className="alert-icon">
                {getEventIcon(alert.type)}
              </div>
              <div className="alert-title">
                <h3>{alert.title}</h3>
                <div className="event-type">{eventTypes[alert.type]?.label || 'Weather Event'}</div>
              </div>
              <div className="severity-badge">
                {threatLevels[alert.severity]?.icon} {threatLevels[alert.severity]?.label || 'Alert'}
              </div>
            </div>

            <div className="alert-details">
              <div className="description">{alert.description}</div>
              <div className="location">
                <FaMapMarkerAlt className="location-icon" />
                <span>{alert.location}</span>
              </div>
              <div className="timeframe">
                <FaClock className="time-icon" />
                <span>{alert.timeframe}</span>
              </div>
            </div>

            {/* Expanded Details Section */}
            {expandedDetails.has(alert.id) && (
              <motion.div
                className="expanded-details"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
                style={{
                  background: 'rgba(0, 0, 0, 0.2)',
                  borderRadius: '8px',
                  padding: '1rem',
                  marginTop: '1rem',
                  border: '1px solid rgba(255, 255, 255, 0.1)'
                }}
              >
                <h4 style={{ color: theme.colors.text.primary, marginBottom: '0.5rem' }}>
                  📋 Detailed Information
                </h4>
                <div style={{ color: theme.colors.text.secondary, fontSize: '0.9rem', lineHeight: '1.6' }}>
                  <p><strong>Alert ID:</strong> #{alert.id}</p>
                  <p><strong>Region:</strong> {alert.region}</p>
                  <p><strong>Event Type:</strong> {eventTypes[alert.type]?.label || 'Weather Event'}</p>
                  <p><strong>Severity Level:</strong> {threatLevels[alert.severity]?.label || 'Alert'}</p>
                  <p><strong>Full Description:</strong> {alert.description}</p>
                  <p><strong>Affected Location:</strong> {alert.location}</p>
                  <p><strong>Time Frame:</strong> {alert.timeframe}</p>
                  <p><strong>Last Updated:</strong> {alert.timestamp}</p>
                  {trackedAlerts.has(alert.id) && (
                    <div style={{ 
                      background: 'rgba(16, 185, 129, 0.2)', 
                      padding: '0.5rem', 
                      borderRadius: '4px', 
                      marginTop: '0.5rem',
                      border: '1px solid rgba(16, 185, 129, 0.3)'
                    }}>
                      <strong>🔍 Status:</strong> This alert is being tracked. You'll receive updates when the situation changes.
                    </div>
                  )}
                </div>
              </motion.div>
            )}

            <div className="alert-footer">
              <div className="alert-time">{alert.timestamp}</div>
              <div className="alert-actions">
                <button 
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('🔥 Details button clicked for alert:', alert.id);
                    setExpandedDetails(prev => {
                      const newSet = new Set(prev);
                      if (newSet.has(alert.id)) {
                        newSet.delete(alert.id);
                        console.log('📄 Hiding details for alert', alert.id);
                      } else {
                        newSet.add(alert.id);
                        console.log('📋 Showing details for alert', alert.id);
                      }
                      return newSet;
                    });
                  }}
                  style={{
                    background: expandedDetails.has(alert.id) ? 
                      'linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%)' : 
                      'rgba(255, 255, 255, 0.1)',
                    color: expandedDetails.has(alert.id) ? 'white' : 'inherit'
                  }}
                >
                  {expandedDetails.has(alert.id) ? '📄 Hide Details' : '📋 Details'}
                </button>
                <button 
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('🔥 Track button clicked for alert:', alert.id);
                    setTrackedAlerts(prev => {
                      const newSet = new Set(prev);
                      if (newSet.has(alert.id)) {
                        newSet.delete(alert.id);
                        console.log('⏹️ Stopped tracking alert', alert.id);
                        window.alert('⏹️ Stopped tracking this alert');
                      } else {
                        newSet.add(alert.id);
                        console.log('🔍 Started tracking alert', alert.id);
                        window.alert(`🔍 Started tracking: ${alert.title}`);
                      }
                      return newSet;
                    });
                  }}
                  style={{
                    background: trackedAlerts.has(alert.id) ? 
                      'linear-gradient(135deg, #10b981 0%, #059669 100%)' : 
                      'rgba(255, 255, 255, 0.1)',
                    color: trackedAlerts.has(alert.id) ? 'white' : 'inherit'
                  }}
                >
                  {trackedAlerts.has(alert.id) ? '⏹️ Stop Tracking' : '📍 Track'}
                </button>
              </div>
            </div>
          </AlertCard>
        ))}
      </AlertsGrid>

      {filteredAlerts.length === 0 && (
        <motion.div
          style={{
            textAlign: 'center',
            padding: '3rem',
            color: theme.colors.text.secondary
          }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6 }}
        >
          <FaExclamationTriangle style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.5 }} />
          <h3>No alerts match your current filters</h3>
          <p>Try adjusting your filter criteria to see more results</p>
        </motion.div>
      )}
    </AlertsContainer>
  );
}

export default GlobalAlerts;