import React from 'react';
import styled from 'styled-components';
import { FaGithub, FaLinkedin, FaEnvelope, FaHeart, FaBrain } from 'react-icons/fa';

const FooterContainer = styled.footer`
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(10px);
  color: white;
  padding: 2rem 0 1rem;
  margin-top: 3rem;
`;

const FooterContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
  display: grid;
  grid-template-columns: 2fr 1fr 1fr;
  gap: 2rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    text-align: center;
  }
`;

const FooterSection = styled.div`
  h3 {
    margin-bottom: 1rem;
    color: #667eea;
    font-size: 1.2rem;
  }
  
  p, li {
    color: rgba(255, 255, 255, 0.8);
    line-height: 1.6;
    margin-bottom: 0.5rem;
  }
  
  ul {
    list-style: none;
    padding: 0;
  }
  
  a {
    color: rgba(255, 255, 255, 0.8);
    text-decoration: none;
    transition: color 0.3s ease;
    
    &:hover {
      color: #667eea;
    }
  }
`;

const SocialLinks = styled.div`
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
  
  @media (max-width: 768px) {
    justify-content: center;
  }
  
  a {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: rgba(102, 126, 234, 0.2);
    border-radius: 50%;
    color: white;
    font-size: 1.2rem;
    transition: all 0.3s ease;
    
    &:hover {
      background: #667eea;
      transform: translateY(-2px);
    }
  }
`;

const FooterBottom = styled.div`
  border-top: 1px solid rgba(255, 255, 255, 0.2);
  margin-top: 2rem;
  padding-top: 1rem;
  text-align: center;
  color: rgba(255, 255, 255, 0.6);
`;

const Copyright = styled.p`
  margin: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  
  svg {
    color: #F44336;
    animation: pulse 2s infinite;
  }
  
  @keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
  }
`;

const TechBadge = styled.span`
  display: inline-block;
  background: rgba(102, 126, 234, 0.2);
  color: #667eea;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.8rem;
  margin: 0.25rem 0.25rem 0.25rem 0;
`;

function Footer() {
  const currentYear = new Date().getFullYear();
  
  return (
    <FooterContainer>
      <FooterContent>
        <FooterSection>
          <h3>Climate Anomaly Detection System</h3>
          <p>
            Advanced climate monitoring platform powered by Bayesian Deep Learning, 
            achieving 97.8% accuracy in real-time anomaly detection and prediction.
          </p>
          <p>
            <strong>Key Technologies:</strong>
          </p>
          <div>
            <TechBadge>Bayesian Neural Networks</TechBadge>
            <TechBadge>Monte Carlo Dropout</TechBadge>
            <TechBadge>Deep Belief Networks</TechBadge>
            <TechBadge>Markov Models</TechBadge>
            <TechBadge>React.js</TechBadge>
            <TechBadge>ERA5 Data</TechBadge>
          </div>
          <SocialLinks>
            <a href="https://github.com" aria-label="GitHub">
              <FaGithub />
            </a>
            <a href="https://linkedin.com" aria-label="LinkedIn">
              <FaLinkedin />
            </a>
            <a href="mailto:team@climate-anomaly.edu" aria-label="Email">
              <FaEnvelope />
            </a>
          </SocialLinks>
        </FooterSection>
        
        <FooterSection>
          <h3>Quick Links</h3>
          <ul>
            <li><a href="/">Dashboard</a></li>
            <li><a href="/monitor">Real-time Monitor</a></li>
            <li><a href="/predictions">Model Predictions</a></li>
            <li><a href="/historical">Historical Data</a></li>
            <li><a href="/about">About Project</a></li>
          </ul>
        </FooterSection>
        
        <FooterSection>
          <h3>Project Info</h3>
          <ul>
            <li>3rd Year Computer Science</li>
            <li>Probability & Statistics Course</li>
            <li>Semester 5 - 2024/2025</li>
            <li>Bayesian Deep Learning Focus</li>
            <li>Open Source Project</li>
          </ul>
        </FooterSection>
      </FooterContent>
      
      <FooterBottom>
        <Copyright>
          Made with <FaHeart /> using <FaBrain /> Bayesian Deep Learning
        </Copyright>
        <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.9rem' }}>
          © {currentYear} Climate Anomaly Detection System. 
          Educational project for academic purposes.
        </p>
        <p style={{ margin: '0.5rem 0 0 0', fontSize: '0.8rem' }}>
          Achieving 97.8% Accuracy • Real-time Monitoring • Uncertainty Quantification
        </p>
      </FooterBottom>
    </FooterContainer>
  );
}

export default Footer;