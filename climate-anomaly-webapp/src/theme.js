// Aura Weather Theme Configuration
export const theme = {
  colors: {
    primary: {
      50: '#f0f9ff',
      100: '#e0f2fe', 
      200: '#bae6fd',
      300: '#7dd3fc',
      400: '#38bdf8',
      500: '#0ea5e9',
      600: '#0284c7',
      700: '#0369a1',
      800: '#075985',
      900: '#0c4a6e'
    },
    secondary: {
      50: '#f8fafc',
      100: '#f1f5f9',
      200: '#e2e8f0',
      300: '#cbd5e1',
      400: '#94a3b8',
      500: '#64748b',
      600: '#475569',
      700: '#334155',
      800: '#1e293b',
      900: '#0f172a'
    },
    accent: {
      warning: '#fbbf24',
      danger: '#ef4444',
      success: '#10b981',
      info: '#60a5fa'
    },
    background: {
      primary: 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
      secondary: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
      card: 'rgba(255, 255, 255, 0.1)',
      cardHover: 'rgba(255, 255, 255, 0.15)'
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.8)',
      muted: 'rgba(255, 255, 255, 0.6)',
      dark: '#1e293b'
    },
    border: {
      primary: 'rgba(255, 255, 255, 0.2)',
      secondary: 'rgba(255, 255, 255, 0.1)'
    }
  },
  glassmorphism: {
    light: {
      background: 'rgba(255, 255, 255, 0.1)',
      backdropFilter: 'blur(20px)',
      border: '1px solid rgba(255, 255, 255, 0.2)',
      boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)'
    },
    medium: {
      background: 'rgba(255, 255, 255, 0.15)',
      backdropFilter: 'blur(25px)',
      border: '1px solid rgba(255, 255, 255, 0.3)',
      boxShadow: '0 25px 50px rgba(0, 0, 0, 0.15)'
    },
    heavy: {
      background: 'rgba(255, 255, 255, 0.2)',
      backdropFilter: 'blur(30px)',
      border: '1px solid rgba(255, 255, 255, 0.4)',
      boxShadow: '0 30px 60px rgba(0, 0, 0, 0.2)'
    }
  },
  typography: {
    fontFamily: "'Inter', 'Segoe UI', 'Roboto', sans-serif",
    fontSize: {
      xs: '0.75rem',
      sm: '0.875rem',
      base: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      '2xl': '1.5rem',
      '3xl': '1.875rem',
      '4xl': '2.25rem',
      '5xl': '3rem'
    },
    fontWeight: {
      light: 300,
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700
    }
  },
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    '2xl': '3rem',
    '3xl': '4rem'
  },
  borderRadius: {
    sm: '8px',
    md: '12px',
    lg: '16px',
    xl: '20px',
    full: '50%'
  },
  animation: {
    duration: {
      fast: '0.2s',
      normal: '0.3s',
      slow: '0.5s'
    },
    easing: {
      default: 'cubic-bezier(0.4, 0, 0.2, 1)',
      in: 'cubic-bezier(0.4, 0, 1, 1)',
      out: 'cubic-bezier(0, 0, 0.2, 1)',
      inOut: 'cubic-bezier(0.4, 0, 0.2, 1)'
    }
  },
  breakpoints: {
    sm: '640px',
    md: '768px',
    lg: '1024px',
    xl: '1280px',
    '2xl': '1536px'
  },
  zIndex: {
    dropdown: 1000,
    modal: 1050,
    tooltip: 1100,
    notification: 1200
  }
};

// Threat Level Color System
export const threatLevels = {
  safe: {
    color: '#10b981',
    background: 'rgba(16, 185, 129, 0.1)',
    label: 'Safe',
    icon: '✅'
  },
  low: {
    color: '#22c55e', 
    background: 'rgba(34, 197, 94, 0.1)',
    label: 'Low Risk',
    icon: '🟢'
  },
  moderate: {
    color: '#fbbf24',
    background: 'rgba(251, 191, 36, 0.1)',
    label: 'Moderate Risk',
    icon: '🟡'
  },
  high: {
    color: '#f97316',
    background: 'rgba(249, 115, 22, 0.1)',
    label: 'High Risk',
    icon: '🟠'
  },
  severe: {
    color: '#ef4444',
    background: 'rgba(239, 68, 68, 0.1)',
    label: 'Severe Danger',
    icon: '🔴'
  },
  extreme: {
    color: '#dc2626',
    background: 'rgba(220, 38, 38, 0.1)',
    label: 'Extreme Danger',
    icon: '🆘'
  }
};

// Weather Icons Mapping
export const weatherIcons = {
  sunny: '☀️',
  partlyCloudy: '⛅',
  cloudy: '☁️',
  rain: '🌧️',
  heavyRain: '⛈️',
  snow: '🌨️',
  fog: '🌫️',
  wind: '💨',
  tornado: '🌪️',
  hurricane: '🌀',
  hail: '🧊',
  thunderstorm: '⚡'
};