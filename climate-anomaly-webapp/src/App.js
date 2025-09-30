import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import styled from 'styled-components';
import Login from './components/Login';
import Header from './components/AuraWeatherHeader';
import Dashboard from './components/AuraWeatherDashboard';
import GlobalAlerts from './components/GlobalAlerts';
import PersonalizedBriefings from './components/PersonalizedBriefings';
import RealTimeMonitor from './components/RealTimeMonitor';
import ModelPredictions from './components/ModelPredictionsIntegrated';
import HistoricalData from './components/HistoricalData';
import About from './components/About';
import Footer from './components/Footer';
import { theme } from './theme';

// Fallback theme in case of import issues (matches theme structure)
const fallbackTheme = {
  colors: {
    background: {
      primary: 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)'
    }
  }
};

// Resolve background safely even if theme import fails or structure changes
const backgroundPrimary = theme?.colors?.background?.primary ?? fallbackTheme.colors.background.primary;

// Lightweight runtime visibility in dev
if (process.env.NODE_ENV !== 'production') {
  // eslint-disable-next-line no-console
  console.log('[AuraWeather] theme resolved:', !!theme, 'bg:', backgroundPrimary);
}

const AppContainer = styled.div`
  min-height: 100vh;
  background: ${() => backgroundPrimary};
`;

const MainContent = styled.main`
  min-height: calc(100vh - 140px);
  padding-top: 80px;
`;

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Initialize app data
  useEffect(() => {
    console.log('�️ Aura Weather - Predictive Weather Intelligence Initialized');
    // Check for existing authentication
    const savedAuth = localStorage.getItem('aura-weather-auth');
    if (savedAuth) {
      setIsAuthenticated(true);
    }
  }, []);

  const handleLogin = (authStatus) => {
    setIsAuthenticated(authStatus);
    if (authStatus) {
      localStorage.setItem('aura-weather-auth', 'true');
    }
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    localStorage.removeItem('aura-weather-auth');
  };

  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <Router
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true
      }}
    >
      <AppContainer>
        <Header onLogout={handleLogout} />
        <MainContent>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/alerts" element={<GlobalAlerts />} />
            <Route path="/briefings" element={<PersonalizedBriefings />} />
            <Route path="/monitor" element={<RealTimeMonitor />} />
            <Route path="/predictions" element={<ModelPredictions />} />
            <Route path="/historical" element={<HistoricalData />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </MainContent>
        <Footer />
      </AppContainer>
    </Router>
  );
}

export default App;