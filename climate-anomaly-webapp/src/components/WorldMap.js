import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';
import L from 'leaflet';
import styled from 'styled-components';

// Fix for default marker icons in React-Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

const MapWrapper = styled.div`
  height: 400px;
  width: 100%;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  
  .leaflet-container {
    height: 100%;
    width: 100%;
  }
`;

const Legend = styled.div`
  margin-top: 1rem;
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  
  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
    
    .legend-color {
      width: 16px;
      height: 16px;
      border-radius: 50%;
    }
  }
`;

// Sample anomaly data points
const anomalyData = [
  {
    id: 1,
    position: [40.7128, -74.0060], // New York
    type: 'temperature',
    severity: 'warning',
    value: '+5.2°C above normal',
    description: 'Unusual temperature spike detected'
  },
  {
    id: 2,
    position: [51.5074, -0.1278], // London
    type: 'precipitation',
    severity: 'critical',
    value: '300% above average',
    description: 'Extreme rainfall event'
  },
  {
    id: 3,
    position: [35.6762, 139.6503], // Tokyo
    type: 'pressure',
    severity: 'normal',
    value: 'Within normal range',
    description: 'Normal atmospheric conditions'
  },
  {
    id: 4,
    position: [-33.8688, 151.2093], // Sydney
    type: 'humidity',
    severity: 'warning',
    value: '20% below normal',
    description: 'Unusually dry conditions'
  },
  {
    id: 5,
    position: [55.7558, 37.6176], // Moscow
    type: 'temperature',
    severity: 'critical',
    value: '-8.5°C below normal',
    description: 'Severe cold anomaly'
  }
];

const getMarkerColor = (severity) => {
  switch(severity) {
    case 'critical': return '#F44336';
    case 'warning': return '#FF9800';
    default: return '#4CAF50';
  }
};

const getCircleRadius = (severity) => {
  switch(severity) {
    case 'critical': return 50000;
    case 'warning': return 30000;
    default: return 15000;
  }
};

function WorldMap() {
  return (
    <>
      <MapWrapper>
        <MapContainer
          center={[20, 0]}
          zoom={2}
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          
          {anomalyData.map((anomaly) => (
            <React.Fragment key={anomaly.id}>
              <Circle
                center={anomaly.position}
                radius={getCircleRadius(anomaly.severity)}
                color={getMarkerColor(anomaly.severity)}
                fillColor={getMarkerColor(anomaly.severity)}
                fillOpacity={0.2}
                weight={2}
              />
              <Marker position={anomaly.position}>
                <Popup>
                  <div style={{ minWidth: '200px' }}>
                    <h4 style={{ margin: '0 0 8px 0', textTransform: 'capitalize' }}>
                      {anomaly.type} Anomaly
                    </h4>
                    <p style={{ margin: '0 0 4px 0', fontWeight: 'bold' }}>
                      {anomaly.value}
                    </p>
                    <p style={{ margin: '0', fontSize: '0.9rem', color: '#666' }}>
                      {anomaly.description}
                    </p>
                    <div style={{ 
                      marginTop: '8px', 
                      padding: '4px 8px', 
                      borderRadius: '4px',
                      background: getMarkerColor(anomaly.severity),
                      color: 'white',
                      fontSize: '0.8rem',
                      textAlign: 'center',
                      textTransform: 'uppercase',
                      fontWeight: 'bold'
                    }}>
                      {anomaly.severity}
                    </div>
                  </div>
                </Popup>
              </Marker>
            </React.Fragment>
          ))}
        </MapContainer>
      </MapWrapper>
      
      <Legend>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#4CAF50' }}></div>
          <span>Normal Conditions</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#FF9800' }}></div>
          <span>Warning Level</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ background: '#F44336' }}></div>
          <span>Critical Alert</span>
        </div>
      </Legend>
    </>
  );
}

export default WorldMap;