import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import styled from 'styled-components';
import { FaCloud, FaExclamationTriangle, FaHome, FaUser, FaSignOutAlt, FaBars, FaTimes, FaCog } from 'react-icons/fa';
import { theme } from '../theme';

const HeaderContainer = styled.header`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  background: ${theme.glassmorphism.medium.background};
  backdrop-filter: ${theme.glassmorphism.medium.backdropFilter};
  border-bottom: ${theme.glassmorphism.medium.border};
  z-index: 1000;
  padding: 0 2rem;
  height: 80px;
`;

const Nav = styled.nav`
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 100%;
  max-width: 1600px;
  margin: 0 auto;
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  font-size: 1.5rem;
  font-weight: ${theme.typography.fontWeight.light};
  color: ${theme.colors.text.primary};
  text-decoration: none;
  
  svg {
    margin-right: 0.75rem;
    color: ${theme.colors.accent.info};
    font-size: 1.8rem;
  }
  
  .app-name {
    .main {
      font-weight: ${theme.typography.fontWeight.semibold};
    }
    .tagline {
      font-size: 0.7rem;
      color: ${theme.colors.text.muted};
      display: block;
      line-height: 1;
    }
  }
`;

const CenterNav = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  
  @media (max-width: 768px) {
    display: none;
  }
`;

const AlertsButton = styled(Link)`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.8rem 1.5rem;
  background: linear-gradient(135deg, ${theme.colors.accent.warning} 0%, #f97316 100%);
  color: white;
  text-decoration: none;
  border-radius: ${theme.borderRadius.lg};
  font-weight: ${theme.typography.fontWeight.medium};
  transition: all ${theme.animation.duration.normal} ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(251, 191, 36, 0.4);
  }
  
  .alert-icon {
    animation: pulse 2s infinite;
  }
  
  @keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
  }
`;

const RightNav = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const NavLink = styled(Link)`
  display: flex;
  align-items: center;
  text-decoration: none;
  color: ${theme.colors.text.secondary};
  font-weight: ${theme.typography.fontWeight.medium};
  padding: 0.6rem 1rem;
  border-radius: ${theme.borderRadius.md};
  transition: all ${theme.animation.duration.normal} ease;
  
  svg {
    margin-right: 0.5rem;
  }
  
  &:hover {
    background: rgba(255, 255, 255, 0.1);
    color: ${theme.colors.text.primary};
  }
  
  &.active {
    background: ${theme.colors.accent.info};
    color: white;
  }
  
  @media (max-width: 768px) {
    justify-content: center;
    padding: 1rem;
  }
`;

const UserMenu = styled.div`
  position: relative;
  
  .user-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1rem;
    background: rgba(255, 255, 255, 0.1);
    border: ${theme.glassmorphism.light.border};
    border-radius: ${theme.borderRadius.md};
    color: ${theme.colors.text.primary};
    cursor: pointer;
    transition: all ${theme.animation.duration.normal} ease;
    
    &:hover {
      background: rgba(255, 255, 255, 0.15);
    }
  }
  
  .user-dropdown {
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 0.5rem;
    background: ${theme.glassmorphism.heavy.background};
    backdrop-filter: ${theme.glassmorphism.heavy.backdropFilter};
    border: ${theme.glassmorphism.heavy.border};
    border-radius: ${theme.borderRadius.lg};
    padding: 0.5rem;
    min-width: 180px;
    box-shadow: ${theme.glassmorphism.heavy.boxShadow};
    
    .dropdown-item {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.8rem 1rem;
      color: ${theme.colors.text.secondary};
      text-decoration: none;
      border-radius: ${theme.borderRadius.sm};
      transition: all ${theme.animation.duration.normal} ease;
      cursor: pointer;
      
      &:hover {
        background: rgba(255, 255, 255, 0.1);
        color: ${theme.colors.text.primary};
      }
    }
  }
`;

const MobileToggle = styled.button`
  display: none;
  background: none;
  border: none;
  font-size: 1.5rem;
  color: ${theme.colors.text.primary};
  cursor: pointer;
  
  @media (max-width: 768px) {
    display: block;
  }
`;

const MobileMenu = styled.div`
  display: none;
  
  @media (max-width: 768px) {
    display: ${props => props.$isOpen ? 'block' : 'none'};
    position: fixed;
    top: 80px;
    left: 0;
    right: 0;
    background: ${theme.glassmorphism.heavy.background};
    backdrop-filter: ${theme.glassmorphism.heavy.backdropFilter};
    border-bottom: ${theme.glassmorphism.heavy.border};
    padding: 2rem;
    
    .mobile-nav-links {
      display: flex;
      flex-direction: column;
      gap: 1rem;
      margin-bottom: 2rem;
    }
    
    .mobile-alerts {
      margin-bottom: 2rem;
    }
  }
`;

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
  color: #666;
  
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #4CAF50;
    animation: pulse 2s infinite;
  }
  
  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }
`;

function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  const toggleMenu = () => setIsMenuOpen(!isMenuOpen);

  const isActive = (path) => location.pathname === path;

  return (
    <HeaderContainer>
      <Nav>
        <Logo as={Link} to="/">
          <FaCloud />
          Climate Anomaly Detection
        </Logo>
        
        <NavLinks $isOpen={isMenuOpen}>
          <li>
            <NavLink 
              to="/" 
              className={isActive('/') ? 'active' : ''}
              onClick={() => setIsMenuOpen(false)}
            >
              <FaChartLine />
              Dashboard
            </NavLink>
          </li>
          <li>
            <NavLink 
              to="/monitor" 
              className={isActive('/monitor') ? 'active' : ''}
              onClick={() => setIsMenuOpen(false)}
            >
              <FaEye />
              Live Monitor
            </NavLink>
          </li>
          <li>
            <NavLink 
              to="/predictions" 
              className={isActive('/predictions') ? 'active' : ''}
              onClick={() => setIsMenuOpen(false)}
            >
              <FaCloud />
              Predictions
            </NavLink>
          </li>
          <li>
            <NavLink 
              to="/historical" 
              className={isActive('/historical') ? 'active' : ''}
              onClick={() => setIsMenuOpen(false)}
            >
              <FaHistory />
              Historical
            </NavLink>
          </li>
          <li>
            <NavLink 
              to="/about" 
              className={isActive('/about') ? 'active' : ''}
              onClick={() => setIsMenuOpen(false)}
            >
              <FaInfoCircle />
              About
            </NavLink>
          </li>
        </NavLinks>

        <StatusIndicator className="hide-mobile">
          <div className="status-dot"></div>
          System Online
        </StatusIndicator>

        <MobileToggle onClick={toggleMenu}>
          {isMenuOpen ? <FaTimes /> : <FaBars />}
        </MobileToggle>
      </Nav>
    </HeaderContainer>
  );
}

export default Header;