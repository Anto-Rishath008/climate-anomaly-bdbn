import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import styled from 'styled-components';
import { FaCloud, FaExclamationTriangle, FaHome, FaUser, FaSignOutAlt, FaBars, FaTimes, FaCog, FaBell } from 'react-icons/fa';
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

function Header({ onLogout }) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const location = useLocation();

  const navigation = [
    { path: '/', label: 'Dashboard', icon: FaHome },
    { path: '/briefings', label: 'Briefings', icon: FaBell }
  ];

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      setIsUserMenuOpen(false);
    };

    if (isUserMenuOpen) {
      document.addEventListener('click', handleClickOutside);
    }

    return () => {
      document.removeEventListener('click', handleClickOutside);
    };
  }, [isUserMenuOpen]);

  const handleLogout = () => {
    if (onLogout) {
      onLogout();
    }
    setIsUserMenuOpen(false);
  };

  const handleUserMenuClick = (e) => {
    e.stopPropagation();
    setIsUserMenuOpen(!isUserMenuOpen);
  };

  return (
    <HeaderContainer>
      <Nav>
        <Logo>
          <FaCloud />
          <div className="app-name">
            <div className="main">Aura Weather</div>
            <div className="tagline">Predictive Intelligence</div>
          </div>
        </Logo>

        <CenterNav>
          <AlertsButton to="/alerts">
            <FaExclamationTriangle className="alert-icon" />
            Alerts
          </AlertsButton>
        </CenterNav>

        <RightNav>
          <NavLink
            to="/"
            className={location.pathname === '/' ? 'active' : ''}
          >
            <FaHome />
            Dashboard
          </NavLink>
          
          <NavLink
            to="/briefings"
            className={location.pathname === '/briefings' ? 'active' : ''}
          >
            <FaBell />
            Briefings
          </NavLink>

          <UserMenu>
            <button 
              className="user-button"
              onClick={handleUserMenuClick}
            >
              <FaUser />
              User
            </button>
            
            {isUserMenuOpen && (
              <div className="user-dropdown" onClick={(e) => e.stopPropagation()}>
                <div className="dropdown-item">
                  <FaCog />
                  Settings
                </div>
                <div className="dropdown-item" onClick={handleLogout}>
                  <FaSignOutAlt />
                  Sign Out
                </div>
              </div>
            )}
          </UserMenu>

          <MobileToggle onClick={() => setIsMenuOpen(!isMenuOpen)}>
            {isMenuOpen ? <FaTimes /> : <FaBars />}
          </MobileToggle>
        </RightNav>
      </Nav>

      <MobileMenu $isOpen={isMenuOpen}>
        <div className="mobile-alerts">
          <AlertsButton to="/alerts" onClick={() => setIsMenuOpen(false)}>
            <FaExclamationTriangle className="alert-icon" />
            Global Alerts
          </AlertsButton>
        </div>
        
        <div className="mobile-nav-links">
          {navigation.map(({ path, label, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              className={location.pathname === path ? 'active' : ''}
              onClick={() => setIsMenuOpen(false)}
            >
              <Icon />
              {label}
            </NavLink>
          ))}
        </div>
      </MobileMenu>
    </HeaderContainer>
  );
}

export default Header;