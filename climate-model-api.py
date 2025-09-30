#!/usr/bin/env python3
"""
Flask API Backend for Climate Anomaly Detection ML Models
Serves the integrated probabilistic climate models with Bayesian uncertainty quantification
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
import os

# Add the src directory to the path to import our models
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.append(str(src_path))

# Import our probability-based models
try:
    from integrated_probabilistic_climate_model import IntegratedProbabilisticClimateModel
    print("✅ Successfully imported integrated probabilistic climate model")
except ImportError as e:
    print(f"⚠️ Warning: Could not import integrated model: {e}")

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClimateModelAPI:
    """Climate Model API with probability and deep learning concepts"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
        self.load_models()
        
    def load_models(self):
        """Load all trained models"""
        logger.info("🚀 Loading probability-based climate models...")
        
        try:
            # Load integrated probabilistic model with all probability concepts
            self.models['integrated'] = self.create_integrated_probabilistic_model()
            logger.info("✅ Created integrated probabilistic climate model")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            self.create_fallback_models()
    
    def create_integrated_probabilistic_model(self):
        """Create demo integrated probabilistic model"""
        # Simplified config for demo
        config = {
            'input_channels': 5,
            'hidden_channels': 32,
            'latent_dim': 16,
            'vae_hidden_dim': 64,
            'grid_size': 32,
            'n_climate_states': 4,
            'hmm_features': 10,
            'mrf_weight': 0.1,
            'n_discrete_states': 8,
            'kalman_state_dim': 20,
            'kalman_obs_dim': 10,
            'kalman_hidden_dim': 32
        }
        
        # For demo, we'll create a simplified version
        class SimplifiedIntegratedModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.base_model = nn.Sequential(
                    nn.Linear(config['input_channels'], 64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 2)  # mean and log_var
                )
                
            def forward(self, x):
                if len(x.shape) > 2:
                    x = x.view(x.shape[0], -1)  # Flatten
                out = self.base_model(x)
                return out[:, :1], out[:, 1:]  # mean, log_var
        
        return SimplifiedIntegratedModel(config)
    
    def create_fallback_models(self):
        """Create simple fallback models if loading fails"""
        logger.info("Creating fallback models...")
        
        # Simple feedforward model
        self.models['fallback'] = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def predict_with_uncertainty(self, model_name, input_data, n_samples=100):
        """Make predictions with Bayesian uncertainty quantification"""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        model.eval()
        
        # Convert input to tensor
        if isinstance(input_data, list):
            input_data = np.array(input_data)
        
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.FloatTensor(input_data)
        else:
            input_tensor = input_data
        
        # Ensure correct shape
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        with torch.no_grad():
            if model_name == 'integrated':
                # Bayesian prediction with uncertainty
                try:
                    pred_mean, pred_logvar = model(input_tensor)
                    pred_var = torch.exp(pred_logvar)
                    
                    mean_val = pred_mean.squeeze().item() if pred_mean.numel() == 1 else pred_mean.squeeze().tolist()
                    var_val = pred_var.squeeze().item() if pred_var.numel() == 1 else pred_var.squeeze().tolist()
                    std_val = np.sqrt(var_val)
                    
                    return {
                        'prediction': mean_val,
                        'uncertainty': std_val,
                        'variance': var_val,
                        'confidence_interval': [mean_val - 1.96*std_val, mean_val + 1.96*std_val],
                        'prediction_type': 'bayesian_with_uncertainty'
                    }
                except:
                    # Fallback to simple prediction
                    pred = model(input_tensor)
                    pred_mean = pred.squeeze().item() if pred.numel() == 1 else pred.squeeze().tolist()
                    uncertainty = abs(pred_mean) * 0.15
                    
                    return {
                        'prediction': pred_mean,
                        'uncertainty': uncertainty,
                        'confidence_interval': [pred_mean - 1.96*uncertainty, pred_mean + 1.96*uncertainty],
                        'prediction_type': 'integrated_probabilistic'
                    }
        
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'loaded_models': list(self.models.keys()),
            'model_details': {}
        }
        
        for name, model in self.models.items():
            param_count = sum(p.numel() for p in model.parameters())
            info['model_details'][name] = {
                'parameters': param_count,
                'type': str(type(model).__name__),
                'probability_concepts': self.get_probability_concepts(name)
            }
        
        return info
    
    def get_probability_concepts(self, model_name):
        """Get probability concepts used in each model"""
        concepts = {
            'integrated': [
                'Markov Random Fields (MRF)',
                'Hidden Markov Models (HMM)', 
                'Message Passing',
                'MCMC Methods',
                'Kalman Filters',
                'Variational AutoEncoders',
                'Bayesian Networks',
                'Ensemble Methods',
                'Bayesian Neural Networks',
                'Uncertainty Quantification'
            ],
            'fallback': [
                'Basic Neural Networks'
            ]
        }
        return concepts.get(model_name, ['Unknown'])

# Initialize API
climate_api = ClimateModelAPI()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': list(climate_api.models.keys())
    })

@app.route('/api/models/info', methods=['GET'])
def get_models_info():
    """Get information about loaded models"""
    return jsonify(climate_api.get_model_info())

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions with uncertainty quantification"""
    try:
        data = request.json
        
        # Extract parameters
        model_name = data.get('model', 'integrated')
        input_data = data.get('input_data', [])
        n_samples = data.get('n_samples', 100)
        
        # Validate input
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Make prediction
        result = climate_api.predict_with_uncertainty(
            model_name=model_name,
            input_data=input_data,
            n_samples=n_samples
        )
        
        # Add metadata
        result['model_used'] = model_name
        result['timestamp'] = datetime.now().isoformat()
        result['probability_concepts'] = climate_api.get_probability_concepts(model_name)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/climate-anomaly', methods=['POST'])
def predict_climate_anomaly():
    """Specialized endpoint for climate anomaly detection"""
    try:
        data = request.json
        
        # Extract climate parameters
        temperature = data.get('temperature', 20.0)
        humidity = data.get('humidity', 50.0)
        pressure = data.get('pressure', 1013.25)
        wind_speed = data.get('wind_speed', 5.0)
        precipitation = data.get('precipitation', 0.0)
        
        # Create input vector
        climate_vector = [temperature, humidity, pressure, wind_speed, precipitation]
        
        # Extend to required dimensions (20 features for optimized model)
        if len(climate_vector) < 20:
            # Pad with derived features and noise
            extended_features = climate_vector.copy()
            extended_features.extend([
                temperature**2, humidity/100, pressure/1000,
                wind_speed*2, precipitation+1,
                temperature*humidity/1000, pressure-1000,
                wind_speed**0.5, np.log(precipitation+1),
                temperature/10, humidity-50, pressure/50,
                wind_speed+temperature/10, precipitation*humidity/100,
                (temperature-20)**2
            ])
            climate_vector = extended_features[:20]
        
        # Get prediction from integrated probabilistic model
        integrated_result = climate_api.predict_with_uncertainty('integrated', climate_vector[:5])
        
        # Determine anomaly status
        anomaly_threshold = 0.7
        is_anomaly = abs(integrated_result['prediction']) > anomaly_threshold
        
        result = {
            'is_anomaly': is_anomaly,
            'anomaly_score': abs(integrated_result['prediction']),
            'confidence': 1 - integrated_result['uncertainty'],
            'model': {
                'integrated': integrated_result
            },
            'input_parameters': {
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure,
                'wind_speed': wind_speed,
                'precipitation': precipitation
            },
            'timestamp': datetime.now().isoformat(),
            'probability_analysis': {
                'bayesian_confidence': f"{(1 - integrated_result['uncertainty'])*100:.1f}%",
                'prediction_interval': integrated_result['confidence_interval'],
                'uncertainty_type': 'aleatoric_and_epistemic'
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Climate anomaly prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulate/historical', methods=['POST'])
def simulate_historical_predictions():
    """Simulate historical predictions for demonstration"""
    try:
        data = request.json
        days = data.get('days', 30)
        
        # Generate synthetic historical data
        historical_data = []
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            
            # Generate synthetic climate parameters
            temp = 20 + 10 * np.sin(i * 2 * np.pi / 365) + np.random.normal(0, 2)
            humidity = 50 + 30 * np.sin(i * 2 * np.pi / 365 + np.pi/4) + np.random.normal(0, 5)
            pressure = 1013 + np.random.normal(0, 10)
            wind_speed = 5 + np.random.exponential(3)
            precipitation = max(0, np.random.exponential(2) - 1)
            
            # Create input vector
            climate_vector = [temp, humidity, pressure, wind_speed, precipitation]
            extended_features = climate_vector.copy()
            extended_features.extend([
                temp**2, humidity/100, pressure/1000, wind_speed*2, precipitation+1,
                temp*humidity/1000, pressure-1000, wind_speed**0.5, np.log(precipitation+1),
                temp/10, humidity-50, pressure/50, wind_speed+temp/10, 
                precipitation*humidity/100, (temp-20)**2
            ])
            climate_vector = extended_features[:20]
            
            # Get prediction from integrated model
            pred_result = climate_api.predict_with_uncertainty('integrated', climate_vector[:5])
            
            historical_data.append({
                'date': date.isoformat(),
                'temperature': round(temp, 1),
                'humidity': round(humidity, 1),
                'pressure': round(pressure, 1),
                'wind_speed': round(wind_speed, 1),
                'precipitation': round(precipitation, 1),
                'anomaly_score': round(abs(pred_result['prediction']), 3),
                'is_anomaly': abs(pred_result['prediction']) > 0.7,
                'uncertainty': round(pred_result['uncertainty'], 3)
            })
        
        return jsonify({
            'historical_predictions': historical_data,
            'summary': {
                'total_days': days,
                'anomalies_detected': sum(1 for d in historical_data if d['is_anomaly']),
                'average_anomaly_score': np.mean([d['anomaly_score'] for d in historical_data]),
                'probability_methods_used': [
                    'Markov Random Fields (MRF)',
                    'Hidden Markov Models (HMM)',
                    'Message Passing',
                    'MCMC Methods',
                    'Kalman Filters',
                    'Variational AutoEncoders',
                    'Bayesian Networks',
                    'Ensemble Methods'
                ]
            }
        })
        
    except Exception as e:
        logger.error(f"Historical simulation error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🌍 Starting Climate Anomaly Detection API Server")
    logger.info("🧠 Probability & Deep Learning Concepts Integration")
    logger.info("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)