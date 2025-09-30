#!/usr/bin/env python3
"""
Test script for the Climate ML API
"""

import requests
import json
import time

def test_api():
    base_url = 'http://localhost:5000/api'
    
    print("🧪 Testing Climate ML API Integration")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1️⃣ Testing API Health...")
    try:
        response = requests.get(f'{base_url}/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data['status']}")
            print(f"📅 Timestamp: {data['timestamp']}")
            print(f"🧠 Models loaded: {data['models_loaded']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    # Test 2: Model info
    print("\n2️⃣ Testing Model Information...")
    try:
        response = requests.get(f'{base_url}/models/info', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models available: {data['loaded_models']}")
            for model, details in data['model_details'].items():
                print(f"   🤖 {model}: {details['parameters']:,} parameters")
                print(f"       Concepts: {', '.join(details['probability_concepts'][:3])}...")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info error: {e}")
    
    # Test 3: Climate anomaly prediction
    print("\n3️⃣ Testing Climate Anomaly Prediction...")
    climate_data = {
        "temperature": 25.5,
        "humidity": 65.0,
        "pressure": 1013.25,
        "wind_speed": 10.0,
        "precipitation": 2.5
    }
    
    try:
        response = requests.post(
            f'{base_url}/predict/climate-anomaly',
            json=climate_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful!")
            print(f"   🎯 Anomaly Score: {data['anomaly_score']:.3f}")
            print(f"   🔮 Confidence: {data['confidence']:.3f}")
            print(f"   ⚠️ Is Anomaly: {data['is_anomaly']}")
            print(f"   📊 Bayesian Confidence: {data['probability_analysis']['bayesian_confidence']}")
            print(f"   📈 Uncertainty: ±{data['models']['optimized']['uncertainty']:.3f}")
        else:
            error_data = response.json()
            print(f"❌ Prediction failed: {error_data.get('error', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction error: {e}")
    
    # Test 4: Extreme weather test
    print("\n4️⃣ Testing Extreme Weather Detection...")
    extreme_data = {
        "temperature": 50.0,  # Extreme heat
        "humidity": 15.0,     # Very low humidity
        "pressure": 950.0,    # Low pressure
        "wind_speed": 35.0,   # High wind
        "precipitation": 50.0 # Heavy rain
    }
    
    try:
        response = requests.post(
            f'{base_url}/predict/climate-anomaly',
            json=extreme_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Extreme weather test successful!")
            print(f"   🌡️ Input: T={extreme_data['temperature']}°C, H={extreme_data['humidity']}%")
            print(f"   🎯 Anomaly Score: {data['anomaly_score']:.3f}")
            print(f"   ⚠️ Is Anomaly: {data['is_anomaly']} (Expected: True)")
            print(f"   🔮 Confidence: {data['confidence']:.3f}")
        else:
            print(f"❌ Extreme weather test failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Extreme weather test error: {e}")
    
    # Test 5: Historical simulation
    print("\n5️⃣ Testing Historical Data Simulation...")
    try:
        response = requests.post(
            f'{base_url}/simulate/historical',
            json={"days": 7},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Historical simulation successful!")
            print(f"   📊 Days simulated: {data['summary']['total_days']}")
            print(f"   ⚠️ Anomalies detected: {data['summary']['anomalies_detected']}")
            print(f"   📈 Average anomaly score: {data['summary']['average_anomaly_score']:.3f}")
            print(f"   🧠 Methods used: {', '.join(data['summary']['probability_methods_used'][:2])}...")
        else:
            print(f"❌ Historical simulation failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Historical simulation error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API Integration Test Complete!")
    print("🌐 React app should now be able to connect to ML models")
    print("📱 Visit http://localhost:3000 to see the integrated system")

if __name__ == "__main__":
    test_api()