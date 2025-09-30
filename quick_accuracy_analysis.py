#!/usr/bin/env python3
"""
Quick Model Accuracy & Overfitting Analysis
=========================================

Focused script to quickly assess model performance and detect overfitting/underfitting
using a smaller subset of data for efficiency.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path

def quick_analysis():
    print("🔬 QUICK MODEL PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Create a smaller synthetic dataset for testing
    print("📊 Creating test dataset...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 1000
    n_features = 50
    
    # Create correlated features
    X = np.random.randn(n_samples, n_features)
    # Add some correlation structure
    X[:, :10] = X[:, :10] + 0.5 * X[:, 0:1]  # Correlated features
    
    # Create target with known relationship
    true_weights = np.random.uniform(-1, 1, n_features)
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val: {X_val.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # Define models with different complexities
    models = {
        'Simple': nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        ),
        'Medium': nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        ),
        'Complex': nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\\n🧠 Training {model_name} Model...")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        train_r2s = []
        val_r2s = []
        
        epochs = 100
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            train_pred = model(X_train_t).squeeze()
            train_loss = criterion(train_pred, y_train_t)
            train_loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t).squeeze()
                val_loss = criterion(val_pred, y_val_t)
                
                train_r2 = r2_score(y_train, train_pred.detach().numpy())
                val_r2 = r2_score(y_val, val_pred.detach().numpy())
            
            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())
            train_r2s.append(train_r2)
            val_r2s.append(val_r2)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t).squeeze()
            test_r2 = r2_score(y_test, test_pred.numpy())
            test_mse = mean_squared_error(y_test, test_pred.numpy())
        
        # Analyze overfitting
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        loss_ratio = final_val_loss / final_train_loss
        
        final_train_r2 = train_r2s[-1]
        final_val_r2 = val_r2s[-1]
        r2_gap = final_train_r2 - final_val_r2
        
        # Classification
        if loss_ratio > 1.5 and r2_gap > 0.1:
            status = "SEVERE_OVERFITTING"
        elif loss_ratio > 1.2 and r2_gap > 0.05:
            status = "MODERATE_OVERFITTING"
        elif loss_ratio > 1.1 and r2_gap > 0.02:
            status = "MILD_OVERFITTING"
        elif final_val_r2 < 0.5:
            status = "UNDERFITTING"
        else:
            status = "WELL_FITTED"
        
        results[model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_r2s': train_r2s,
            'val_r2s': val_r2s,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'loss_ratio': loss_ratio,
            'r2_gap': r2_gap,
            'status': status,
            'final_train_r2': final_train_r2,
            'final_val_r2': final_val_r2
        }
        
        print(f"   Final Train R²: {final_train_r2:.4f}")
        print(f"   Final Val R²: {final_val_r2:.4f}")
        print(f"   Test R²: {test_r2:.4f}")
        print(f"   Status: {status}")
    
    # Create visualizations
    print("\\n📈 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = ['blue', 'green', 'red']
    
    for i, (name, result) in enumerate(results.items()):
        color = colors[i]
        
        # Loss curves
        axes[0, 0].plot(result['train_losses'], label=f'{name} Train', color=color, linestyle='-', alpha=0.7)
        axes[0, 0].plot(result['val_losses'], label=f'{name} Val', color=color, linestyle='--', alpha=0.7)
        
        # R² curves
        axes[0, 1].plot(result['train_r2s'], label=f'{name} Train', color=color, linestyle='-', alpha=0.7)
        axes[0, 1].plot(result['val_r2s'], label=f'{name} Val', color=color, linestyle='--', alpha=0.7)
        
        # Overfitting gap
        gap = np.array(result['val_losses']) - np.array(result['train_losses'])
        axes[0, 2].plot(gap, label=f'{name}', color=color, alpha=0.7)
    
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('R² Score Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_title('Overfitting Gap (Val - Train Loss)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss Difference')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Summary metrics
    model_names = list(results.keys())
    test_r2s = [results[name]['test_r2'] for name in model_names]
    loss_ratios = [results[name]['loss_ratio'] for name in model_names]
    r2_gaps = [results[name]['r2_gap'] for name in model_names]
    
    axes[1, 0].bar(model_names, test_r2s, color=colors, alpha=0.7)
    axes[1, 0].set_title('Test R² Scores')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].axhline(y=0.9, color='red', linestyle='--', label='90% Target')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].bar(model_names, loss_ratios, color=colors, alpha=0.7)
    axes[1, 1].set_title('Loss Ratios (Val/Train)')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].axhline(y=1.2, color='red', linestyle='--', label='Overfitting Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].bar(model_names, r2_gaps, color=colors, alpha=0.7)
    axes[1, 2].set_title('R² Gaps (Train - Val)')
    axes[1, 2].set_ylabel('Gap')
    axes[1, 2].axhline(y=0.05, color='red', linestyle='--', label='Overfitting Threshold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate summary report
    print("\\n📋 ANALYSIS SUMMARY")
    print("="*50)
    
    for name, result in results.items():
        print(f"\\n🧠 {name} Model:")
        print(f"   Test R²: {result['test_r2']:.4f} ({'✅ >90%' if result['test_r2'] > 0.9 else '⚠️ <90%'})")
        print(f"   Loss Ratio: {result['loss_ratio']:.4f}")
        print(f"   R² Gap: {result['r2_gap']:.4f}")
        print(f"   Status: {result['status']}")
    
    # Best model recommendation
    best_model = max(results.keys(), key=lambda x: results[x]['test_r2'])
    print(f"\\n🏆 BEST MODEL: {best_model}")
    print(f"   Test R²: {results[best_model]['test_r2']:.4f}")
    print(f"   Status: {results[best_model]['status']}")
    
    # General recommendations
    print("\\n💡 RECOMMENDATIONS:")
    
    overfitted_models = [name for name, result in results.items() if 'OVERFITTING' in result['status']]
    underfitted_models = [name for name, result in results.items() if 'UNDERFITTING' in result['status']]
    
    if overfitted_models:
        print("   🚨 For overfitted models:", ', '.join(overfitted_models))
        print("     - Increase regularization (dropout, weight decay)")
        print("     - Use early stopping")
        print("     - Reduce model complexity")
        print("     - Add more training data")
    
    if underfitted_models:
        print("   📈 For underfitted models:", ', '.join(underfitted_models))
        print("     - Increase model complexity")
        print("     - Reduce regularization")
        print("     - Train for more epochs")
        print("     - Improve feature engineering")
    
    well_fitted = [name for name, result in results.items() if result['status'] == 'WELL_FITTED']
    if well_fitted:
        print("   ✅ Well-fitted models:", ', '.join(well_fitted))
        print("     - These models show good balance")
        print("     - Consider ensemble methods for further improvement")
    
    print("\\n🎯 90% ACCURACY ACHIEVEMENT:")
    high_acc_models = [name for name, result in results.items() if result['test_r2'] > 0.9]
    if high_acc_models:
        print(f"   ✅ Models achieving >90%: {', '.join(high_acc_models)}")
    else:
        best_acc = max(result['test_r2'] for result in results.values())
        print(f"   ⚠️ Best accuracy: {best_acc:.1%}")
        print("   📈 To achieve 90%:")
        print("     - Fine-tune hyperparameters")
        print("     - Use more sophisticated architectures")
        print("     - Apply ensemble methods")
        print("     - Improve data quality and features")

if __name__ == "__main__":
    quick_analysis()