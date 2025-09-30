import React, { useState } from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { FaBrain, FaChartLine, FaCloudUploadAlt } from 'react-icons/fa';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';

const PredictionsContainer = styled.div`
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 3rem;
  
  h1 {
    font-size: 2.5rem;
    color: white;
    margin-bottom: 1rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
  }
  
  p {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.9);
  }
`;

const ModelCard = styled(motion.div)`
  background: rgba(255, 255, 255, 0.95);
  border-radius: 16px;
  padding: 2rem;
  margin-bottom: 2rem;
  box-shadow: 0 8px 32px rgba(0,0,0,0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.2);
`;

const ModelHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  
  h2 {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0;
    color: #333;
    
    svg {
      color: #667eea;
    }
  }
`;

const AccuracyBadge = styled.div`
  padding: 0.5rem 1rem;
  border-radius: 20px;
  background: #4CAF50;
  color: white;
  font-weight: bold;
  font-size: 1.1rem;
`;

const UploadSection = styled.div`
  border: 2px dashed #667eea;
  border-radius: 12px;
  padding: 2rem;
  text-align: center;
  margin-bottom: 2rem;
  transition: all 0.3s ease;
  cursor: pointer;
  
  &:hover {
    border-color: #4CAF50;
    background: rgba(76, 175, 80, 0.05);
  }
  
  svg {
    font-size: 3rem;
    color: #667eea;
    margin-bottom: 1rem;
  }
  
  h3 {
    margin: 0 0 0.5rem 0;
    color: #333;
  }
  
  p {
    margin: 0;
    color: #666;
  }
`;

const ResultsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  
  @media (max-width: 1024px) {
    grid-template-columns: 1fr;
  }
`;

// Sample prediction data
const predictionData = [
  { day: 'Day 1', temperature: 2.3, precipitation: 0.8, pressure: 1.2 },
  { day: 'Day 2', temperature: 3.1, precipitation: 1.5, pressure: 0.9 },
  { day: 'Day 3', temperature: 4.2, precipitation: 2.3, pressure: 1.8 },
  { day: 'Day 4', temperature: 3.8, precipitation: 1.9, pressure: 2.1 },
  { day: 'Day 5', temperature: 2.9, precipitation: 1.2, pressure: 1.5 },
  { day: 'Day 6', temperature: 3.5, precipitation: 1.8, pressure: 1.3 },
  { day: 'Day 7', temperature: 4.1, precipitation: 2.1, pressure: 1.7 }
];

const modelPerformance = [
  { model: 'Bayesian NN', accuracy: 97.8, precision: 96.5, recall: 98.2 },
  { model: 'Deep Belief Net', accuracy: 96.3, precision: 95.1, recall: 97.4 },
  { model: 'Conv U-Net', accuracy: 95.7, precision: 94.8, recall: 96.6 },
  { model: 'Traditional ML', accuracy: 87.2, precision: 86.1, recall: 88.3 }
];

function ModelPredictions() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setIsProcessing(true);
      
      // Simulate processing
      setTimeout(() => {
        setPredictions({
          anomalyProbability: 0.73,
          confidence: 0.94,
          riskLevel: 'High',
          recommendedAction: 'Immediate monitoring required'
        });
        setIsProcessing(false);
      }, 3000);
    }
  };

  return (
    <PredictionsContainer>
      <Header>
        <h1>Model Predictions</h1>
        <p>Advanced climate anomaly predictions using Bayesian Deep Learning</p>
      </Header>

      <ModelCard
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <ModelHeader>
          <h2>
            <FaBrain />
            Bayesian Neural Network
          </h2>
          <AccuracyBadge>97.8% Accuracy</AccuracyBadge>
        </ModelHeader>
        
        <UploadSection onClick={() => document.getElementById('file-upload').click()}>
          <input
            id="file-upload"
            type="file"
            accept=".csv,.json,.npz"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
          <FaCloudUploadAlt />
          <h3>Upload Climate Data for Prediction</h3>
          <p>Support formats: CSV, JSON, NPZ • Max size: 50MB</p>
          {selectedFile && (
            <p style={{ marginTop: '1rem', color: '#4CAF50', fontWeight: 'bold' }}>
              Selected: {selectedFile.name}
            </p>
          )}
        </UploadSection>
        
        {isProcessing && (
          <div style={{ textAlign: 'center', padding: '2rem' }}>
            <div style={{
              width: '50px',
              height: '50px',
              border: '3px solid #f3f3f3',
              borderTop: '3px solid #667eea',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite',
              margin: '0 auto 1rem'
            }}></div>
            <p>Processing data with Bayesian neural network...</p>
          </div>
        )}
        
        {predictions && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{
              padding: '1.5rem',
              background: 'rgba(102, 126, 234, 0.1)',
              borderRadius: '12px',
              marginTop: '1rem'
            }}
          >
            <h3 style={{ margin: '0 0 1rem 0' }}>Prediction Results</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
              <div>
                <strong>Anomaly Probability:</strong><br />
                <span style={{ fontSize: '1.5rem', color: '#F44336' }}>
                  {(predictions.anomalyProbability * 100).toFixed(1)}%
                </span>
              </div>
              <div>
                <strong>Confidence:</strong><br />
                <span style={{ fontSize: '1.5rem', color: '#4CAF50' }}>
                  {(predictions.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div>
                <strong>Risk Level:</strong><br />
                <span style={{ fontSize: '1.2rem', color: '#FF9800', fontWeight: 'bold' }}>
                  {predictions.riskLevel}
                </span>
              </div>
            </div>
            <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(255,255,255,0.8)', borderRadius: '8px' }}>
              <strong>Recommendation:</strong> {predictions.recommendedAction}
            </div>
          </motion.div>
        )}
      </ModelCard>

      <ResultsGrid>
        <ModelCard
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <h2 style={{ marginBottom: '1.5rem' }}>
            <FaChartLine style={{ color: '#667eea', marginRight: '0.5rem' }} />
            7-Day Anomaly Forecast
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={predictionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="day" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="temperature" stroke="#FF6B6B" name="Temperature" />
              <Line type="monotone" dataKey="precipitation" stroke="#4ECDC4" name="Precipitation" />
              <Line type="monotone" dataKey="pressure" stroke="#45B7D1" name="Pressure" />
            </LineChart>
          </ResponsiveContainer>
        </ModelCard>

        <ModelCard
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <h2 style={{ marginBottom: '1.5rem' }}>
            <FaBrain style={{ color: '#667eea', marginRight: '0.5rem' }} />
            Model Performance Comparison
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelPerformance}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="accuracy" fill="#667eea" name="Accuracy (%)" />
              <Bar dataKey="precision" fill="#4CAF50" name="Precision (%)" />
              <Bar dataKey="recall" fill="#FF9800" name="Recall (%)" />
            </BarChart>
          </ResponsiveContainer>
        </ModelCard>
      </ResultsGrid>
    </PredictionsContainer>
  );
}

export default ModelPredictions;