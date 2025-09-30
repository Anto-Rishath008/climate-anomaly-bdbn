import React, { useEffect, useRef, useState } from 'react';
import styled from 'styled-components';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import { theme } from '../theme';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet default markers
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

const MapWrapper = styled.div`
  width: 100%;
  height: 100%;
  min-height: 500px;
  border-radius: ${theme?.borderRadius?.xl || '16px'};
  overflow: hidden;
  position: relative;
  
  .leaflet-container {
    height: 100%;
    width: 100%;
    border-radius: ${theme?.borderRadius?.xl || '16px'};
  }
  
  .leaflet-control-layers {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.2);
  }
  
  .leaflet-popup-content-wrapper {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.2);
  }
`;

const MapOverlay = styled.div`
  position: absolute;
  top: 1rem;
  left: 1rem;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 8px;
  padding: 0.5rem 1rem;
  color: white;
  font-size: 0.9rem;
  pointer-events: none;
  z-index: 1000;
`;

const CoordinateDisplay = styled.div`
  position: absolute;
  bottom: 1rem;
  right: 1rem;
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-family: monospace;
  font-size: 0.85rem;
  z-index: 1000;
  min-width: 200px;
`;

const LayerControl = styled.div`
  position: absolute;
  top: 1rem;
  right: 1rem;
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 8px;
  padding: 0.5rem;
  z-index: 1000;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  
  button {
    padding: 0.4rem 0.8rem;
    border: none;
    border-radius: 4px;
    background: ${theme?.colors?.accent?.info || '#60a5fa'};
    color: white;
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.2s ease;
    
    &:hover {
      background: ${theme?.colors?.primary?.[600] || '#4f46e5'};
      transform: translateY(-1px);
    }
    
    &.active {
      background: ${theme?.colors?.accent?.success || '#10b981'};
    }
  }
`;

// Component to handle map clicks
function MapClickHandler({ onLocationSelect, onCoordinatesChange }) {
  const [position, setPosition] = useState(null);
  
  useMapEvents({
    click: async (e) => {
      const { lat, lng } = e.latlng;
      setPosition([lat, lng]);
      
      // Try to get location name using reverse geocoding with Nominatim (free)
      try {
        const response = await fetch(
          `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&addressdetails=1`
        );
        
        if (response.ok) {
          const data = await response.json();
          const locationName = data.display_name || `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
          
          const newLocation = {
            name: locationName,
            coords: { lat, lng }
          };
          
          onLocationSelect(newLocation);
          onCoordinatesChange && onCoordinatesChange({ lat, lng });
        }
      } catch (error) {
        // Fallback to coordinates display
        const locationName = `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
        const newLocation = {
          name: locationName,
          coords: { lat, lng }
        };
        
        onLocationSelect(newLocation);
        onCoordinatesChange && onCoordinatesChange({ lat, lng });
      }
    },
  });

  return position === null ? null : (
    <Marker position={position}>
      <Popup>
        <div>
          <strong>Selected Location</strong><br />
          Lat: {position[0].toFixed(4)}<br />
          Lng: {position[1].toFixed(4)}
        </div>
      </Popup>
    </Marker>
  );
}

const InteractiveWorldMap = ({ 
  selectedLocation, 
  onLocationSelect, 
  onCoordinatesChange 
}) => {
  const [mapLayer, setMapLayer] = useState('satellite');
  const [mouseCoords, setMouseCoords] = useState({ lat: 0, lng: 0 });
  const mapRef = useRef();

  // Map tile layers - multiple free options
  const tileLayers = {
    satellite: {
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      attribution: '&copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    },
    streets: {
      url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    },
    terrain: {
      url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
      attribution: 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)'
    },
    dark: {
      url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
    }
  };

  const currentLayer = tileLayers[mapLayer];

  // Handle mouse move over map
  const handleMouseMove = (e) => {
    if (e.latlng) {
      setMouseCoords({ lat: e.latlng.lat, lng: e.latlng.lng });
    }
  };

  // Update marker when selectedLocation changes
  useEffect(() => {
    if (selectedLocation && selectedLocation.coords && mapRef.current) {
      const map = mapRef.current;
      map.setView([selectedLocation.coords.lat, selectedLocation.coords.lng], 10, {
        animate: true,
        duration: 1
      });
    }
  }, [selectedLocation]);

  return (
    <MapWrapper>
      <MapOverlay>
        🛰️ Interactive Satellite Map - Click anywhere to select location
      </MapOverlay>
      
      <LayerControl>
        <button 
          className={mapLayer === 'satellite' ? 'active' : ''}
          onClick={() => setMapLayer('satellite')}
        >
          🛰️ Satellite
        </button>
        <button 
          className={mapLayer === 'streets' ? 'active' : ''}
          onClick={() => setMapLayer('streets')}
        >
          🗺️ Streets
        </button>
        <button 
          className={mapLayer === 'terrain' ? 'active' : ''}
          onClick={() => setMapLayer('terrain')}
        >
          🏔️ Terrain
        </button>
        <button 
          className={mapLayer === 'dark' ? 'active' : ''}
          onClick={() => setMapLayer('dark')}
        >
          🌙 Dark
        </button>
      </LayerControl>
      
      <MapContainer
        ref={mapRef}
        center={[20, 0]} // Center on world
        zoom={2}
        style={{ height: '100%', width: '100%' }}
        worldCopyJump={true}
        maxBounds={[[-90, -180], [90, 180]]}
        scrollWheelZoom={true}
        doubleClickZoom={true}
        zoomControl={true}
        attributionControl={true}
        eventHandlers={{
          mousemove: handleMouseMove,
        }}
      >
        <TileLayer
          url={currentLayer.url}
          attribution={currentLayer.attribution}
          maxZoom={18}
          noWrap={false}
        />
        
        <MapClickHandler 
          onLocationSelect={onLocationSelect}
          onCoordinatesChange={onCoordinatesChange}
        />
        
        {/* Show marker for selected location */}
        {selectedLocation && selectedLocation.coords && (
          <Marker position={[selectedLocation.coords.lat, selectedLocation.coords.lng]}>
            <Popup>
              <div>
                <strong>{selectedLocation.name}</strong><br />
                Lat: {selectedLocation.coords.lat.toFixed(4)}<br />
                Lng: {selectedLocation.coords.lng.toFixed(4)}
              </div>
            </Popup>
          </Marker>
        )}
      </MapContainer>
      
      <CoordinateDisplay>
        <div>Mouse: {mouseCoords.lat.toFixed(4)}, {mouseCoords.lng.toFixed(4)}</div>
        {selectedLocation && selectedLocation.coords && (
          <div>
            Selected: {selectedLocation.coords.lat.toFixed(4)}, {selectedLocation.coords.lng.toFixed(4)}
          </div>
        )}
      </CoordinateDisplay>
    </MapWrapper>
  );
};

export default InteractiveWorldMap;