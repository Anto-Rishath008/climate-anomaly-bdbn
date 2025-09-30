#!/usr/bin/env python3
"""
Comprehensive Model Performance Analysis
======================================

This script analyzes model performance, checks for overfitting/underfitting,
and provides detailed accuracy metrics for the climate anomaly detection system.

Key analyses:
1. Training vs Validation Performance
2. Learning Curves Analysis  
3. Bias-Variance Decomposition
4. Overfitting/Underfitting Detection
5. Model Complexity Assessment
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import learning_curve, validation_curve
from scipy import stats
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelPerformanceAnalyzer:
    """
    Comprehensive performance analysis for climate models
    """
    
    def __init__(self):
        self.results = {}
        self.figures_dir = Path("model_analysis_figures")
        self.figures_dir.mkdir(exist_ok=True)
        
    def load_dataset(self):
        """Load the 90% accuracy dataset"""
        try:
            data = np.load("data_advanced/advanced_climate_dataset_90acc.npz")
            X_train = data['train_features']
            y_train = data['train_targets'] 
            X_val = data['val_features']
            y_val = data['val_targets']
            X_test = data['test_features']
            y_test = data['test_targets']
            
            print(f"✅ Dataset loaded:")
            print(f"   Train: {X_train.shape[0]} samples")
            print(f"   Validation: {X_val.shape[0]} samples") 
            print(f"   Test: {X_test.shape[0]} samples")
            
            return X_train, y_train, X_val, y_val, X_test, y_test
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
            
    def create_enhanced_model(self, input_shape, dropout_rate=0.3):
        """
        Create enhanced model with regularization for overfitting prevention
        """
        class EnhancedClimateModel(nn.Module):
            def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.3):
                super().__init__()
                
                layers = []
                prev_dim = input_dim
                
                for i, hidden_dim in enumerate(hidden_dims):
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    prev_dim = hidden_dim
                
                # Output layer (no dropout after final layer)
                layers.append(nn.Linear(prev_dim, 1))
                
                self.network = nn.Sequential(*layers)
                
                # Initialize weights
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)
                        
            def forward(self, x):
                return self.network(x)
        
        # Calculate input dimension
        if len(input_shape) == 4:  # (batch, time, height, width)
            input_dim = np.prod(input_shape[1:])
        elif len(input_shape) == 3:  # (batch, time, features)
            input_dim = np.prod(input_shape[1:])
        else:
            input_dim = input_shape[-1]
            
        return EnhancedClimateModel(input_dim, dropout=dropout_rate)
    
    def train_with_monitoring(self, model, train_loader, val_loader, epochs=100, lr=0.001):
        """
        Train model with comprehensive monitoring
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True
        )
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        train_r2s = []
        val_r2s = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = 20
        
        print(f"🏋️ Training model for up to {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_preds, train_targets = [], []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.reshape(batch_X.size(0), -1)  # Flatten
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(outputs.squeeze().detach().numpy())
                train_targets.extend(batch_y.numpy())
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_preds, val_targets = [], []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.reshape(batch_X.size(0), -1)
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    
                    val_loss += loss.item()
                    val_preds.extend(outputs.squeeze().numpy())
                    val_targets.extend(batch_y.numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_r2 = r2_score(train_targets, train_preds)
            val_r2 = r2_score(val_targets, val_preds)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_r2s.append(train_r2)
            val_r2s.append(val_r2)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model_performance.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, "
                      f"Train R²={train_r2:.4f}, Val R²={val_r2:.4f}")
            
            if patience_counter >= early_stop_patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
        
        # Load best model
        model.load_state_dict(torch.load('best_model_performance.pth'))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_r2s': train_r2s,
            'val_r2s': val_r2s,
            'best_epoch': len(train_losses) - patience_counter,
            'best_val_loss': best_val_loss
        }
    
    def analyze_overfitting(self, history):
        """
        Comprehensive overfitting analysis
        """
        train_losses = history['train_losses']
        val_losses = history['val_losses']
        train_r2s = history['train_r2s']
        val_r2s = history['val_r2s']
        
        analysis = {}
        
        # 1. Loss gap analysis
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        loss_gap = final_val_loss - final_train_loss
        loss_ratio = final_val_loss / final_train_loss
        
        # 2. R² gap analysis  
        final_train_r2 = train_r2s[-1]
        final_val_r2 = val_r2s[-1]
        r2_gap = final_train_r2 - final_val_r2
        
        # 3. Best validation vs final
        best_val_loss = min(val_losses)
        best_val_idx = val_losses.index(best_val_loss)
        degradation = final_val_loss - best_val_loss
        
        # 4. Learning curve trend analysis
        if len(val_losses) >= 20:
            recent_val_trend = np.polyfit(range(-10, 0), val_losses[-10:], 1)[0]
        else:
            recent_val_trend = 0
        
        # Classification
        if loss_ratio > 1.5 and r2_gap > 0.1:
            overfitting_status = "SEVERE_OVERFITTING"
        elif loss_ratio > 1.2 and r2_gap > 0.05:
            overfitting_status = "MODERATE_OVERFITTING"  
        elif loss_ratio > 1.1 and r2_gap > 0.02:
            overfitting_status = "MILD_OVERFITTING"
        elif final_val_r2 < 0.5:
            overfitting_status = "UNDERFITTING"
        else:
            overfitting_status = "WELL_FITTED"
            
        analysis = {
            'overfitting_status': overfitting_status,
            'loss_gap': loss_gap,
            'loss_ratio': loss_ratio,
            'r2_gap': r2_gap,
            'final_train_r2': final_train_r2,
            'final_val_r2': final_val_r2,
            'best_val_loss': best_val_loss,
            'degradation': degradation,
            'recent_trend': recent_val_trend
        }
        
        return analysis
    
    def evaluate_model(self, model, test_loader):
        """
        Comprehensive model evaluation
        """
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.reshape(batch_X.size(0), -1)
                outputs = model(batch_X)
                predictions.extend(outputs.squeeze().numpy())
                targets.extend(batch_y.numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate comprehensive metrics
        r2 = r2_score(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        
        # Correlation coefficient
        correlation = np.corrcoef(targets, predictions)[0, 1]
        
        # Explained variance
        explained_var = 1 - np.var(targets - predictions) / np.var(targets)
        
        return {
            'r2_score': r2,
            'correlation': correlation,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'explained_variance': explained_var,
            'predictions': predictions,
            'targets': targets
        }
    
    def create_visualizations(self, history, analysis, evaluation):
        """
        Create comprehensive visualizations
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Learning curves
        axes[0, 0].plot(history['train_losses'], label='Training Loss', alpha=0.8)
        axes[0, 0].plot(history['val_losses'], label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Learning Curves (Loss)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. R² curves
        axes[0, 1].plot(history['train_r2s'], label='Training R²', alpha=0.8)
        axes[0, 1].plot(history['val_r2s'], label='Validation R²', alpha=0.8)
        axes[0, 1].set_title('R² Score Progress')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Prediction scatter plot
        axes[0, 2].scatter(evaluation['targets'], evaluation['predictions'], alpha=0.6)
        axes[0, 2].plot([evaluation['targets'].min(), evaluation['targets'].max()], 
                       [evaluation['targets'].min(), evaluation['targets'].max()], 'r--', lw=2)
        axes[0, 2].set_title(f'Predictions vs Targets\\nR² = {evaluation["r2_score"]:.4f}')
        axes[0, 2].set_xlabel('True Values')
        axes[0, 2].set_ylabel('Predictions')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Residuals plot
        residuals = evaluation['targets'] - evaluation['predictions']
        axes[1, 0].scatter(evaluation['predictions'], residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title('Residuals Plot')
        axes[1, 0].set_xlabel('Predictions')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Residuals histogram
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Overfitting analysis
        train_val_gap = np.array(history['val_losses']) - np.array(history['train_losses'])
        axes[1, 2].plot(train_val_gap, label='Val - Train Loss', alpha=0.8)
        axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 2].set_title('Overfitting Monitor\\n(Val - Train Loss)')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss Difference')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'comprehensive_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_report(self, analysis, evaluation, history):
        """
        Generate comprehensive performance report
        """
        report = f"""
# 📊 COMPREHENSIVE MODEL PERFORMANCE ANALYSIS

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 ACCURACY ASSESSMENT

### Core Performance Metrics
- **R² Score:** {evaluation['r2_score']:.4f} ({'✅ EXCELLENT' if evaluation['r2_score'] > 0.9 else '⚠️ NEEDS IMPROVEMENT' if evaluation['r2_score'] > 0.7 else '❌ POOR'})
- **Correlation:** {evaluation['correlation']:.4f} ({'✅ STRONG' if evaluation['correlation'] > 0.9 else '⚠️ MODERATE' if evaluation['correlation'] > 0.7 else '❌ WEAK'})
- **RMSE:** {evaluation['rmse']:.6f}
- **MAE:** {evaluation['mae']:.6f}
- **Explained Variance:** {evaluation['explained_variance']:.4f}

### 90% Accuracy Target
- **Status:** {'✅ ACHIEVED' if evaluation['r2_score'] >= 0.9 else '❌ NOT ACHIEVED'}
- **Performance:** {evaluation['r2_score']*100:.2f}% vs 90% target

## 🔍 OVERFITTING/UNDERFITTING ANALYSIS

### Overall Assessment: **{analysis['overfitting_status']}**

### Detailed Analysis
- **Loss Ratio (Val/Train):** {analysis['loss_ratio']:.4f}
  - {'✅ Good' if analysis['loss_ratio'] < 1.2 else '⚠️ Concerning' if analysis['loss_ratio'] < 1.5 else '❌ Severe overfitting'}
- **R² Gap (Train - Val):** {analysis['r2_gap']:.4f}
  - {'✅ Good' if analysis['r2_gap'] < 0.05 else '⚠️ Concerning' if analysis['r2_gap'] < 0.1 else '❌ Severe overfitting'}
- **Training R²:** {analysis['final_train_r2']:.4f}
- **Validation R²:** {analysis['final_val_r2']:.4f}

### Training Dynamics
- **Best Validation Loss:** {analysis['best_val_loss']:.6f}
- **Performance Degradation:** {analysis['degradation']:.6f}
- **Recent Trend:** {'📈 Improving' if analysis['recent_trend'] < -0.001 else '📉 Degrading' if analysis['recent_trend'] > 0.001 else '➡️ Stable'}

## 📈 TRAINING SUMMARY

- **Total Epochs:** {len(history['train_losses'])}
- **Best Epoch:** {history['best_epoch']}
- **Final Training Loss:** {history['train_losses'][-1]:.6f}
- **Final Validation Loss:** {history['val_losses'][-1]:.6f}

## 🎯 RECOMMENDATIONS

### If Overfitting Detected:
1. **Increase Regularization:** Higher dropout rates, weight decay
2. **Early Stopping:** Stop training at best validation performance
3. **Data Augmentation:** Increase training data diversity
4. **Model Simplification:** Reduce model complexity
5. **Cross-Validation:** Use k-fold validation for robustness

### If Underfitting Detected:
1. **Increase Model Complexity:** More layers/neurons
2. **Reduce Regularization:** Lower dropout rates
3. **Feature Engineering:** Add more informative features
4. **Longer Training:** Increase number of epochs
5. **Learning Rate Tuning:** Optimize learning rate schedule

### For Optimal Performance:
1. **Ensemble Methods:** Combine multiple models
2. **Hyperparameter Optimization:** Grid/random search
3. **Advanced Architectures:** Try attention mechanisms
4. **Domain Knowledge:** Incorporate climate science insights

---

*Analysis performed by Comprehensive Model Performance Analyzer*
*Climate Anomaly Detection System*
"""
        
        # Save report
        report_path = self.figures_dir / 'model_performance_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report

def main():
    print("🔬 COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    analyzer = ModelPerformanceAnalyzer()
    
    # Load dataset
    print("\n📊 Loading Dataset...")
    dataset = analyzer.load_dataset()
    if dataset is None:
        print("❌ Cannot proceed without dataset")
        return
        
    X_train, y_train, X_val, y_val, X_test, y_test = dataset
    
    # Convert to tensors and create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create and train model
    print("\n🧠 Creating Enhanced Model...")
    model = analyzer.create_enhanced_model(X_train.shape)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train with monitoring
    print("\n🏋️ Training with Comprehensive Monitoring...")
    history = analyzer.train_with_monitoring(model, train_loader, val_loader, epochs=100)
    
    # Analyze overfitting
    print("\n🔍 Analyzing Overfitting/Underfitting...")
    analysis = analyzer.analyze_overfitting(history)
    print(f"Overfitting Status: {analysis['overfitting_status']}")
    
    # Evaluate model
    print("\n📊 Final Model Evaluation...")
    evaluation = analyzer.evaluate_model(model, test_loader)
    print(f"Test R² Score: {evaluation['r2_score']:.4f}")
    print(f"Test Correlation: {evaluation['correlation']:.4f}")
    
    # Create visualizations
    print("\n📈 Creating Visualizations...")
    analyzer.create_visualizations(history, analysis, evaluation)
    
    # Generate report
    print("\n📋 Generating Comprehensive Report...")
    report = analyzer.generate_report(analysis, evaluation, history)
    
    print("\n🎉 ANALYSIS COMPLETE!")
    print(f"📊 Results saved to: {analyzer.figures_dir}")
    print(f"📈 Visualizations: comprehensive_model_analysis.png")
    print(f"📋 Report: model_performance_report.md")

if __name__ == "__main__":
    main()