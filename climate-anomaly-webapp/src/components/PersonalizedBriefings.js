import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { 
  FaLeaf, 
  FaAnchor, 
  FaTruck, 
  FaCalendarAlt,
  FaHistory,
  FaFlask,
  FaUsers,
  FaBell,
  FaCog,
  FaCheck,
  FaTimes,
  FaChevronRight,
  FaPlay,
  FaPause
} from 'react-icons/fa';
import { theme, threatLevels } from '../theme';

const BriefingsContainer = styled.div`
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
`;

const BriefingsHeader = styled.div`
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
      color: ${theme.colors.accent.info};
    }
  }
  
  p {
    color: ${theme.colors.text.secondary};
    font-size: 1.1rem;
  }
`;

const TabsContainer = styled.div`
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
  
  .tab {
    padding: 1rem 2rem;
    background: ${props => props.activeTab === props.tabId ? 
      `linear-gradient(135deg, ${theme.colors.accent.info} 0%, ${theme.colors.primary[600]} 100%)` :
      theme.glassmorphism.light.background
    };
    backdrop-filter: ${theme.glassmorphism.light.backdropFilter};
    border: ${theme.glassmorphism.light.border};
    border-radius: ${theme.borderRadius.lg};
    color: ${props => props.activeTab === props.tabId ? 'white' : theme.colors.text.secondary};
    cursor: pointer;
    transition: all ${theme.animation.duration.normal} ease;
    
    &:hover {
      background: ${props => props.activeTab === props.tabId ? 
        `linear-gradient(135deg, ${theme.colors.accent.info} 0%, ${theme.colors.primary[600]} 100%)` :
        theme.glassmorphism.medium.background
      };
      color: ${props => props.activeTab === props.tabId ? 'white' : theme.colors.text.primary};
    }
  }
`;

const SubscriptionChannels = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 2rem;
  margin-bottom: 3rem;
`;

const ChannelCard = styled(motion.div)`
  background: ${theme.glassmorphism.medium.background};
  backdrop-filter: ${theme.glassmorphism.medium.backdropFilter};
  border: ${theme.glassmorphism.medium.border};
  border-radius: ${theme.borderRadius.xl};
  padding: 2rem;
  position: relative;
  overflow: hidden;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: ${props => props.color || theme.colors.accent.info};
  }
  
  .channel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    
    .channel-info {
      display: flex;
      align-items: center;
      gap: 1rem;
      
      .channel-icon {
        font-size: 2rem;
        color: ${props => props.color || theme.colors.accent.info};
      }
      
      .channel-details {
        h3 {
          color: ${theme.colors.text.primary};
          font-size: 1.3rem;
          margin: 0 0 0.5rem 0;
        }
        
        .channel-type {
          color: ${theme.colors.text.muted};
          font-size: 0.9rem;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }
      }
    }
    
    .subscription-toggle {
      position: relative;
      width: 60px;
      height: 30px;
      border-radius: 15px;
      cursor: pointer;
      transition: all ${theme.animation.duration.normal} ease;
      
      &::after {
        content: '';
        position: absolute;
        top: 3px;
        width: 24px;
        height: 24px;
        background: white;
        border-radius: 50%;
        transition: all ${theme.animation.duration.normal} ease;
      }
    }
    
    &[data-subscribed="true"] .subscription-toggle {
      background: linear-gradient(135deg, ${theme.colors.accent.success} 0%, #059669 100%);
      
      &::after {
        left: 33px;
      }
    }
    
    &[data-subscribed="false"] .subscription-toggle {
      background: rgba(255, 255, 255, 0.2);
      
      &::after {
        left: 3px;
      }
    }
  }
  
  .channel-description {
    color: ${theme.colors.text.secondary};
    line-height: 1.6;
    margin-bottom: 1.5rem;
  }
  
  .channel-examples {
    .examples-title {
      color: ${theme.colors.text.primary};
      font-weight: ${theme.typography.fontWeight.medium};
      margin-bottom: 0.5rem;
      font-size: 0.9rem;
    }
    
    .example-item {
      color: ${theme.colors.text.muted};
      font-size: 0.85rem;
      margin-bottom: 0.25rem;
      padding-left: 1rem;
      position: relative;
      
      &::before {
        content: '•';
        position: absolute;
        left: 0;
        color: ${props => props.color || theme.colors.accent.info};
      }
    }
  }
`;

const InnovativeFeatures = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 2rem;
`;

const FeatureCard = styled(motion.div)`
  background: ${theme.glassmorphism.light.background};
  backdrop-filter: ${theme.glassmorphism.light.backdropFilter};
  border: ${theme.glassmorphism.light.border};
  border-radius: ${theme.borderRadius.xl};
  padding: 2rem;
  
  .feature-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    
    .feature-icon {
      font-size: 2rem;
      color: ${theme.colors.accent.info};
    }
    
    h3 {
      color: ${theme.colors.text.primary};
      font-size: 1.3rem;
      margin: 0;
    }
  }
  
  .feature-description {
    color: ${theme.colors.text.secondary};
    line-height: 1.6;
    margin-bottom: 1.5rem;
  }
  
  .feature-action {
    padding: 0.8rem 1.5rem;
    background: linear-gradient(135deg, ${theme.colors.accent.info} 0%, ${theme.colors.primary[600]} 100%);
    border: none;
    border-radius: ${theme.borderRadius.md};
    color: white;
    font-weight: ${theme.typography.fontWeight.medium};
    cursor: pointer;
    transition: all ${theme.animation.duration.normal} ease;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    
    &:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 20px rgba(96, 165, 250, 0.3);
    }
  }
`;

const AlertsDemo = styled.div`
  background: ${theme.glassmorphism.medium.background};
  backdrop-filter: ${theme.glassmorphism.medium.backdropFilter};
  border: ${theme.glassmorphism.medium.border};
  border-radius: ${theme.borderRadius.lg};
  padding: 1.5rem;
  margin-top: 1rem;
  
  .demo-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    
    h4 {
      color: ${theme.colors.text.primary};
      margin: 0;
    }
    
    .demo-controls {
      display: flex;
      gap: 0.5rem;
      
      button {
        padding: 0.4rem 0.8rem;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: ${theme.borderRadius.sm};
        color: ${theme.colors.text.secondary};
        cursor: pointer;
        transition: all ${theme.animation.duration.normal} ease;
        
        &:hover {
          background: rgba(255, 255, 255, 0.15);
          color: ${theme.colors.text.primary};
        }
      }
    }
  }
  
  .alert-item {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: ${theme.borderRadius.md};
    margin-bottom: 0.5rem;
    
    .alert-icon {
      font-size: 1.2rem;
      color: ${theme.colors.accent.warning};
      margin-top: 0.2rem;
    }
    
    .alert-content {
      flex: 1;
      
      .alert-text {
        color: ${theme.colors.text.primary};
        font-size: 0.9rem;
        line-height: 1.5;
      }
      
      .alert-time {
        color: ${theme.colors.text.muted};
        font-size: 0.8rem;
        margin-top: 0.25rem;
      }
    }
  }
`;

function PersonalizedBriefings() {
  const [activeTab, setActiveTab] = useState('channels');
  const [subscriptions, setSubscriptions] = useState({
    agriculture: true,
    marine: false,
    logistics: true,
    events: false
  });
  
  const [demoAlerts, setDemoAlerts] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);

  const channels = [
    {
      id: 'agriculture',
      name: 'Agriculture',
      icon: FaLeaf,
      color: '#10b981',
      type: 'Farming & Crops',
      description: 'Specialized weather alerts for farmers and agricultural professionals. Get early warnings about conditions that could affect crops, livestock, and farming operations.',
      examples: [
        'High chance of hailstorms this week in your region. Consider protecting your crops.',
        'Drought conditions expected next month. Plan irrigation schedules.',
        'Optimal planting conditions forecast for next weekend.',
        'Frost warning: Temperatures dropping below 0°C tonight.'
      ]
    },
    {
      id: 'marine',
      name: 'Marine & Fishing',
      icon: FaAnchor,
      color: '#0ea5e9',
      type: 'Maritime Operations',
      description: 'Critical weather information for maritime activities, fishing operations, and coastal businesses. Stay safe on the water with precise marine forecasts.',
      examples: [
        'Unusually strong offshore winds predicted for Friday. Avoid small craft navigation.',
        'Perfect fishing conditions expected: Calm seas and optimal visibility.',
        'Storm surge warning: Coastal flooding possible in low-lying areas.',
        'Fog bank approaching: Visibility dropping to less than 1 nautical mile.'
      ]
    },
    {
      id: 'logistics',
      name: 'Logistics & Transport',
      icon: FaTruck,
      color: '#f59e0b',
      type: 'Supply Chain',
      description: 'Weather intelligence for transportation, shipping, and logistics companies. Optimize routes and schedules with weather-aware planning.',
      examples: [
        'Risk of heavy fog on major highways tomorrow morning. Expect delays.',
        'Clear weather window for next 72 hours - optimal for long-haul transport.',
        'Ice storm approaching I-95 corridor. Consider alternative routes.',
        'High winds affecting bridge operations - trucks may be restricted.'
      ]
    },
    {
      id: 'events',
      name: 'Event Planning',
      icon: FaCalendarAlt,
      color: '#8b5cf6',
      type: 'Outdoor Events',
      description: 'Weather planning assistance for event organizers, wedding planners, and outdoor venue managers. Make informed decisions about outdoor activities.',
      examples: [
        'High probability of rain on Saturday afternoon. Recommend moving your outdoor event indoors.',
        'Perfect weather for outdoor events: Clear skies and comfortable temperatures.',
        'Strong winds expected during evening hours - secure all outdoor decorations.',
        'Heat advisory in effect - provide extra hydration stations for guests.'
      ]
    }
  ];

  const innovativeFeatures = [
    {
      id: 'historical',
      name: 'Historical Anomaly Explorer',
      icon: FaHistory,
      description: 'Explore past weather anomalies and disasters for any location. Perfect for researchers, students, and curious minds wanting to understand climate patterns.',
      action: 'Explore Timeline'
    },
    {
      id: 'community',
      name: 'Community Reporting',
      icon: FaUsers,
      description: 'Help validate predictions by reporting actual conditions in your area. Contribute to improving model accuracy through crowd-sourced verification.',
      action: 'Join Community'
    },
    {
      id: 'scenarios',
      name: 'What-If Scenarios',
      icon: FaFlask,
      description: 'Advanced feature to explore potential impacts by tweaking parameters. See how temperature or humidity changes could affect local weather patterns.',
      action: 'Try Scenarios'
    }
  ];

  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        const sampleAlerts = [
          { text: '🌾 Drought conditions developing in Northern California. Consider water conservation measures.', time: 'Just now', icon: FaLeaf },
          { text: '⛵ Small craft advisory issued for Pacific Coast. Winds 25-35 knots expected.', time: '2 min ago', icon: FaAnchor },
          { text: '🚛 Fog advisory for I-5 corridor. Visibility reduced to 1/4 mile in sections.', time: '5 min ago', icon: FaTruck },
          { text: '🎪 Perfect weather window for outdoor events this weekend. Clear skies forecast.', time: '8 min ago', icon: FaCalendarAlt }
        ];
        
        const randomAlert = sampleAlerts[Math.floor(Math.random() * sampleAlerts.length)];
        setDemoAlerts(prev => [randomAlert, ...prev.slice(0, 4)]);
      }, 3000);

      return () => clearInterval(interval);
    }
  }, [isPlaying]);

  const toggleSubscription = (channelId) => {
    setSubscriptions(prev => ({
      ...prev,
      [channelId]: !prev[channelId]
    }));
  };

  const toggleDemo = () => {
    setIsPlaying(!isPlaying);
    if (!isPlaying) {
      setDemoAlerts([]);
    }
  };

  const getChannelColor = (channelId) => {
    const channel = channels.find(c => c.id === channelId);
    return channel?.color || theme.colors.accent.info;
  };

  return (
    <BriefingsContainer>
      <BriefingsHeader>
        <h1>
          <FaBell className="header-icon" />
          Personalized Briefings
        </h1>
        <p>Subscribe to specialized alert channels and explore innovative weather intelligence features</p>
      </BriefingsHeader>

      <TabsContainer activeTab={activeTab}>
        <div 
          className="tab" 
          data-tab-id="channels"
          onClick={() => setActiveTab('channels')}
        >
          Subscription Channels
        </div>
        <div 
          className="tab" 
          data-tab-id="features"
          onClick={() => setActiveTab('features')}
        >
          Innovative Features
        </div>
      </TabsContainer>

      {activeTab === 'channels' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <SubscriptionChannels>
            {channels.map((channel, index) => (
              <ChannelCard
                key={channel.id}
                color={channel.color}
                data-subscribed={subscriptions[channel.id].toString()}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ y: -5, transition: { duration: 0.2 } }}
              >
                <div className="channel-header">
                  <div className="channel-info">
                    <channel.icon className="channel-icon" />
                    <div className="channel-details">
                      <h3>{channel.name}</h3>
                      <div className="channel-type">{channel.type}</div>
                    </div>
                  </div>
                  <div 
                    className="subscription-toggle"
                    onClick={() => toggleSubscription(channel.id)}
                  />
                </div>
                
                <div className="channel-description">
                  {channel.description}
                </div>
                
                <div className="channel-examples">
                  <div className="examples-title">Example Alerts:</div>
                  {channel.examples.map((example, idx) => (
                    <div key={idx} className="example-item">
                      {example}
                    </div>
                  ))}
                </div>
              </ChannelCard>
            ))}
          </SubscriptionChannels>

          <AlertsDemo>
            <div className="demo-header">
              <h4>Live Alert Simulation</h4>
              <div className="demo-controls">
                <button onClick={toggleDemo}>
                  {isPlaying ? <FaPause /> : <FaPlay />}
                  {isPlaying ? 'Pause' : 'Start'} Demo
                </button>
              </div>
            </div>
            
            {demoAlerts.map((alert, index) => (
              <motion.div
                key={index}
                className="alert-item"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5 }}
              >
                <alert.icon className="alert-icon" />
                <div className="alert-content">
                  <div className="alert-text">{alert.text}</div>
                  <div className="alert-time">{alert.time}</div>
                </div>
              </motion.div>
            ))}
            
            {!isPlaying && demoAlerts.length === 0 && (
              <div style={{ textAlign: 'center', color: theme.colors.text.muted, padding: '2rem' }}>
                Click "Start Demo" to see live alerts simulation
              </div>
            )}
          </AlertsDemo>
        </motion.div>
      )}

      {activeTab === 'features' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <InnovativeFeatures>
            {innovativeFeatures.map((feature, index) => (
              <FeatureCard
                key={feature.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ y: -5, transition: { duration: 0.2 } }}
              >
                <div className="feature-header">
                  <feature.icon className="feature-icon" />
                  <h3>{feature.name}</h3>
                </div>
                
                <div className="feature-description">
                  {feature.description}
                </div>
                
                <button className="feature-action">
                  {feature.action}
                  <FaChevronRight />
                </button>
              </FeatureCard>
            ))}
          </InnovativeFeatures>
        </motion.div>
      )}
    </BriefingsContainer>
  );
}

export default PersonalizedBriefings;