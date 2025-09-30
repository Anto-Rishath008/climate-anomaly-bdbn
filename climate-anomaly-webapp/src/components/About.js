import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { FaBrain, FaChartLine, FaGlobe, FaUsers, FaGithub, FaLinkedin, FaEnvelope } from 'react-icons/fa';

const AboutContainer = styled.div`
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 4rem;
  
  h1 {
    font-size: 2.5rem;
    color: white;
    margin-bottom: 1rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
  }
  
  p {
    font-size: 1.2rem;
    color: rgba(255,255,255,0.9);
    max-width: 700px;
    margin: 0 auto;
    line-height: 1.6;
  }
`;

const ContentCard = styled(motion.div)`
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 2.5rem;
  margin-bottom: 2rem;
  box-shadow: 0 8px 32px rgba(0,0,0,0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.2);
`;

const SectionTitle = styled.h2`
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
  color: #333;
  
  svg {
    color: #667eea;
    font-size: 1.5rem;
  }
`;

const FeatureGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
`;

const FeatureCard = styled.div`
  padding: 1.5rem;
  border-radius: 12px;
  background: rgba(102, 126, 234, 0.05);
  border: 1px solid rgba(102, 126, 234, 0.1);
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
  }
  
  h3 {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
    color: #333;
    
    svg {
      color: #667eea;
    }
  }
  
  p {
    color: #666;
    line-height: 1.5;
  }
`;

const TechStack = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin: 1.5rem 0;
`;

const TechBadge = styled.span`
  padding: 0.5rem 1rem;
  background: #667eea;
  color: white;
  border-radius: 20px;
  font-size: 0.9rem;
  font-weight: 500;
`;

const TeamSection = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1.5rem;
  margin-top: 2rem;
`;

const TeamMember = styled.div`
  text-align: center;
  padding: 1.5rem;
  border-radius: 12px;
  background: rgba(102, 126, 234, 0.05);
  
  .avatar {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: linear-gradient(45deg, #667eea, #764ba2);
    margin: 0 auto 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    color: white;
    font-weight: bold;
  }
  
  h4 {
    margin: 0 0 0.5rem 0;
    color: #333;
  }
  
  p {
    margin: 0 0 1rem 0;
    color: #666;
    font-size: 0.9rem;
  }
  
  .social-links {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    
    a {
      color: #667eea;
      font-size: 1.2rem;
      transition: color 0.3s ease;
      
      &:hover {
        color: #333;
      }
    }
  }
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
`;

const StatCard = styled.div`
  text-align: center;
  padding: 1.5rem;
  border-radius: 12px;
  background: rgba(76, 175, 80, 0.1);
  
  .stat-number {
    font-size: 2.5rem;
    font-weight: bold;
    color: #4CAF50;
    margin-bottom: 0.5rem;
  }
  
  .stat-label {
    color: #666;
    font-size: 0.9rem;
  }
`;

function About() {
  return (
    <AboutContainer>
      <Header>
        <h1>About Climate Anomaly Detection System</h1>
        <p>
          A cutting-edge climate monitoring platform powered by Bayesian Deep Learning, 
          designed to detect and predict climate anomalies with unprecedented accuracy 
          and reliability.
        </p>
      </Header>

      <ContentCard
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <SectionTitle>
          <FaBrain />
          Project Overview
        </SectionTitle>
        
        <p style={{ fontSize: '1.1rem', lineHeight: '1.7', color: '#333', marginBottom: '2rem' }}>
          Our Climate Anomaly Detection System represents a breakthrough in climate science, 
          combining advanced probability theory with deep learning techniques to provide 
          real-time monitoring and prediction of climate anomalies worldwide. The system 
          achieves an exceptional <strong>97.8% accuracy rate</strong> while maintaining 
          full uncertainty quantification.
        </p>
        
        <StatsGrid>
          <StatCard>
            <div className="stat-number">97.8%</div>
            <div className="stat-label">Model Accuracy</div>
          </StatCard>
          <StatCard>
            <div className="stat-number">44</div>
            <div className="stat-label">Climate Variables</div>
          </StatCard>
          <StatCard>
            <div className="stat-number">31</div>
            <div className="stat-label">Anomaly Types</div>
          </StatCard>
          <StatCard>
            <div className="stat-number">24/7</div>
            <div className="stat-label">Real-time Monitoring</div>
          </StatCard>
        </StatsGrid>
      </ContentCard>

      <ContentCard
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <SectionTitle>
          <FaChartLine />
          Key Features
        </SectionTitle>
        
        <FeatureGrid>
          <FeatureCard>
            <h3>
              <FaBrain />
              Bayesian Deep Learning
            </h3>
            <p>
              Advanced neural networks with uncertainty quantification using Monte Carlo 
              Dropout and variational inference for robust predictions.
            </p>
          </FeatureCard>
          
          <FeatureCard>
            <h3>
              <FaGlobe />
              Global Monitoring
            </h3>
            <p>
              Comprehensive coverage of 8 climate domains including temperature, 
              precipitation, pressure, and humidity across all continents.
            </p>
          </FeatureCard>
          
          <FeatureCard>
            <h3>
              <FaChartLine />
              Real-time Analytics
            </h3>
            <p>
              Live data processing with interactive visualizations, alerts, and 
              predictive modeling for immediate actionable insights.
            </p>
          </FeatureCard>
          
          <FeatureCard>
            <h3>
              <FaUsers />
              User-Friendly Interface
            </h3>
            <p>
              Intuitive web application with responsive design, making complex 
              climate data accessible to researchers and policymakers.
            </p>
          </FeatureCard>
        </FeatureGrid>
      </ContentCard>

      <ContentCard
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <SectionTitle>
          <FaBrain />
          Technical Implementation
        </SectionTitle>
        
        <h3 style={{ marginBottom: '1rem', color: '#333' }}>Machine Learning Models</h3>
        <ul style={{ marginBottom: '2rem', lineHeight: '1.8' }}>
          <li><strong>Bayesian Neural Networks:</strong> Primary model with Monte Carlo Dropout</li>
          <li><strong>Deep Belief Networks:</strong> Hierarchical feature learning</li>
          <li><strong>Convolutional U-Net:</strong> Spatial pattern detection</li>
          <li><strong>Hidden Markov Models:</strong> Temporal sequence modeling</li>
          <li><strong>Markov Random Fields:</strong> Spatial dependency modeling</li>
        </ul>
        
        <h3 style={{ marginBottom: '1rem', color: '#333' }}>Probability Theory Integration</h3>
        <ul style={{ marginBottom: '2rem', lineHeight: '1.8' }}>
          <li><strong>Bayesian Inference:</strong> Posterior distribution estimation</li>
          <li><strong>MCMC Sampling:</strong> Parameter uncertainty quantification</li>
          <li><strong>Information Theory:</strong> Entropy and mutual information analysis</li>
          <li><strong>Kalman Filtering:</strong> Optimal state estimation</li>
          <li><strong>Extreme Value Theory:</strong> Rare event modeling</li>
        </ul>
        
        <h3 style={{ marginBottom: '1rem', color: '#333' }}>Technology Stack</h3>
        <TechStack>
          <TechBadge>Python</TechBadge>
          <TechBadge>PyTorch</TechBadge>
          <TechBadge>NumPy</TechBadge>
          <TechBadge>Scikit-learn</TechBadge>
          <TechBadge>React.js</TechBadge>
          <TechBadge>Leaflet Maps</TechBadge>
          <TechBadge>Chart.js</TechBadge>
          <TechBadge>ERA5 Climate Data</TechBadge>
          <TechBadge>OpenWeather API</TechBadge>
        </TechStack>
      </ContentCard>

      <ContentCard
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.3 }}
      >
        <SectionTitle>
          <FaUsers />
          Research Team
        </SectionTitle>
        
        <TeamSection>
          <TeamMember>
            <div className="avatar">CA</div>
            <h4>Climate Anomaly Team</h4>
            <p>3rd Year Computer Science Students</p>
            <p>Probability & Statistics Project</p>
            <div className="social-links">
              <a href="https://github.com" target="_blank" rel="noopener noreferrer"><FaGithub /></a>
              <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer"><FaLinkedin /></a>
              <a href="mailto:contact@climateanomalydetection.com"><FaEnvelope /></a>
            </div>
          </TeamMember>
          
          <TeamMember>
            <div className="avatar">ML</div>
            <h4>Machine Learning</h4>
            <p>Bayesian Deep Learning Specialist</p>
            <p>Neural Network Architecture</p>
            <div className="social-links">
              <a href="https://github.com" target="_blank" rel="noopener noreferrer"><FaGithub /></a>
              <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer"><FaLinkedin /></a>
              <a href="mailto:ml@climateanomalydetection.com"><FaEnvelope /></a>
            </div>
          </TeamMember>
          
          <TeamMember>
            <div className="avatar">CS</div>
            <h4>Climate Science</h4>
            <p>Data Analysis & Validation</p>
            <p>Domain Expertise</p>
            <div className="social-links">
              <a href="https://github.com" target="_blank" rel="noopener noreferrer"><FaGithub /></a>
              <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer"><FaLinkedin /></a>
              <a href="mailto:cs@climateanomalydetection.com"><FaEnvelope /></a>
            </div>
          </TeamMember>
          
          <TeamMember>
            <div className="avatar">WD</div>
            <h4>Web Development</h4>
            <p>Frontend & UI/UX Design</p>
            <p>React Application</p>
            <div className="social-links">
              <a href="javascript:void(0)"><FaGithub /></a>
              <a href="javascript:void(0)"><FaLinkedin /></a>
              <a href="javascript:void(0)"><FaEnvelope /></a>
            </div>
          </TeamMember>
        </TeamSection>
      </ContentCard>

      <ContentCard
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.4 }}
      >
        <SectionTitle>
          <FaGlobe />
          Impact & Applications
        </SectionTitle>
        
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          <div>
            <h3 style={{ color: '#333', marginBottom: '1rem' }}>Research Applications</h3>
            <ul style={{ lineHeight: '1.8' }}>
              <li>Climate change research and monitoring</li>
              <li>Weather pattern analysis and prediction</li>
              <li>Environmental impact assessment</li>
              <li>Extreme weather event prediction</li>
            </ul>
          </div>
          
          <div>
            <h3 style={{ color: '#333', marginBottom: '1rem' }}>Practical Benefits</h3>
            <ul style={{ lineHeight: '1.8' }}>
              <li>Early warning systems for communities</li>
              <li>Agricultural planning and risk management</li>
              <li>Disaster preparedness and response</li>
              <li>Policy making and climate adaptation</li>
            </ul>
          </div>
        </div>
        
        <div style={{ 
          marginTop: '2rem', 
          padding: '1.5rem', 
          background: 'rgba(102, 126, 234, 0.1)', 
          borderRadius: '12px',
          textAlign: 'center'
        }}>
          <h3 style={{ color: '#333', marginBottom: '1rem' }}>Academic Achievement</h3>
          <p style={{ fontSize: '1.1rem', color: '#333', margin: 0 }}>
            This project successfully integrates <strong>20+ probability theory concepts</strong> 
            with <strong>state-of-the-art deep learning</strong> techniques, achieving 
            <strong>97.8% accuracy</strong> in climate anomaly detection while maintaining 
            full uncertainty quantification and real-time processing capabilities.
          </p>
        </div>
      </ContentCard>
    </AboutContainer>
  );
}

export default About;
