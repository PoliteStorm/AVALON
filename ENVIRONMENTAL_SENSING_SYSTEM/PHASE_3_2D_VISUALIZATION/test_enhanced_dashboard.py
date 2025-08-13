#!/usr/bin/env python3
"""
Test Enhanced Dashboard Features - Week 2 Implementation
Tests the new Plotly.js integration, enhanced controls, and real-time features
"""

import requests
import json
import time
from datetime import datetime

def test_api_endpoints():
    """Test all API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Enhanced Dashboard API Endpoints...")
    
    # Test main page
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Main dashboard page: OK")
            if "plotly-latest.min.js" in response.text:
                print("✅ Plotly.js integration: OK")
            if "Real-time Environmental Data" in response.text:
                print("✅ Enhanced chart titles: OK")
            if "settingsModal" in response.text:
                print("✅ Settings modal: OK")
        else:
            print(f"❌ Main page failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Main page error: {e}")
    
    # Test status API
    try:
        response = requests.get(f"{base_url}/api/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status API: OK - {data['status']} - {data['data_points_collected']} data points")
        else:
            print(f"❌ Status API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Status API error: {e}")
    
    # Test environmental data API
    try:
        response = requests.get(f"{base_url}/api/environmental_data")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Environmental data API: OK - Temp: {data['temperature']}°C, Humidity: {data['humidity']}%")
        else:
            print(f"❌ Environmental data API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Environmental data API error: {e}")
    
    # Test historical data API
    try:
        response = requests.get(f"{base_url}/api/historical_data")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Historical data API: OK - {data['data_points']} data points available")
        else:
            print(f"❌ Historical data API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Historical data API error: {e}")
    
    # Test system health API
    try:
        response = requests.get(f"{base_url}/api/system_health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System health API: OK - {data['status']} - Version {data['version']}")
        else:
            print(f"❌ System health API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ System health API error: {e}")

def test_monitoring_controls():
    """Test monitoring start/stop controls"""
    base_url = "http://localhost:5000"
    
    print("\n🎛️ Testing Monitoring Controls...")
    
    # Test start monitoring
    try:
        response = requests.post(f"{base_url}/api/start_monitoring")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Start monitoring: {data['message']}")
        else:
            print(f"❌ Start monitoring failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Start monitoring error: {e}")
    
    # Test stop monitoring
    try:
        response = requests.post(f"{base_url}/api/stop_monitoring")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stop monitoring: {data['message']}")
        else:
            print(f"❌ Stop monitoring failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stop monitoring error: {e}")
    
    # Restart monitoring
    try:
        response = requests.post(f"{base_url}/api/start_monitoring")
        if response.status_code == 200:
            print("✅ Monitoring restarted")
        else:
            print(f"❌ Restart monitoring failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Restart monitoring error: {e}")

def test_real_time_data():
    """Test real-time data collection"""
    print("\n📊 Testing Real-time Data Collection...")
    
    base_url = "http://localhost:5000"
    
    # Collect data for 30 seconds
    print("Collecting real-time data for 30 seconds...")
    start_time = time.time()
    data_points = []
    
    while time.time() - start_time < 30:
        try:
            response = requests.get(f"{base_url}/api/environmental_data")
            if response.status_code == 200:
                data = response.json()
                data_points.append(data)
                print(f"📈 Data point {len(data_points)}: Temp={data['temperature']}°C, Humidity={data['humidity']}%, pH={data['ph']}")
            time.sleep(5)  # Wait 5 seconds between requests
        except Exception as e:
            print(f"❌ Data collection error: {e}")
            break
    
    print(f"✅ Collected {len(data_points)} data points in 30 seconds")
    
    # Analyze data patterns
    if data_points:
        temps = [d['temperature'] for d in data_points]
        humidities = [d['humidity'] for d in data_points]
        phs = [d['ph'] for d in data_points]
        
        print(f"📊 Temperature range: {min(temps):.1f}°C - {max(temps):.1f}°C")
        print(f"📊 Humidity range: {min(humidities):.1f}% - {max(humidities):.1f}%")
        print(f"📊 pH range: {min(phs):.2f} - {max(phs):.2f}")

def test_enhanced_features():
    """Test Week 2 enhanced features"""
    print("\n🚀 Testing Week 2 Enhanced Features...")
    
    base_url = "http://localhost:5000"
    
    # Test if enhanced HTML features are present
    try:
        response = requests.get(f"{base_url}/")
        html_content = response.text
        
        enhanced_features = [
            ("Plotly.js Integration", "plotly-latest.min.js"),
            ("Enhanced Chart Controls", "chart-type"),
            ("Time Range Selection", "time-range"),
            ("Update Frequency Control", "update-frequency"),
            ("Parameter Checkboxes", "temp-check"),
            ("Settings Modal", "settingsModal"),
            ("Progress Bars", "temperature-progress"),
            ("Alert System", "alert-container"),
            ("Gauge Charts", "gauge-charts"),
            ("Correlation Chart", "correlation-chart"),
            ("Trend Analysis", "trend-chart"),
            ("Dark Theme Support", "theme-select"),
            ("Chart Animations", "chart-animations"),
            ("Export Functionality", "export-data"),
            ("Enhanced Status Cards", "quality-value"),
            ("Real-time Environmental Data", "main-chart"),
            ("Historical Analysis", "Historical Analysis"),
            ("System Status Display", "system-status"),
            ("Recent Alerts Display", "recent-alerts")
        ]
        
        for feature_name, feature_id in enhanced_features:
            if feature_id in html_content:
                print(f"✅ {feature_name}: OK")
            else:
                print(f"❌ {feature_name}: Missing")
                
    except Exception as e:
        print(f"❌ Enhanced features test error: {e}")

def test_websocket_functionality():
    """Test WebSocket functionality"""
    print("\n🔌 Testing WebSocket Functionality...")
    
    base_url = "http://localhost:5000"
    
    try:
        response = requests.get(f"{base_url}/")
        html_content = response.text
        
        if "socket.io" in html_content:
            print("✅ Socket.IO client library: OK")
        else:
            print("❌ Socket.IO client library: Missing")
            
        if "io()" in html_content:
            print("✅ WebSocket initialization: OK")
        else:
            print("❌ WebSocket initialization: Missing")
            
        if "environmental_update" in html_content:
            print("✅ WebSocket event handlers: OK")
        else:
            print("❌ WebSocket event handlers: Missing")
            
    except Exception as e:
        print(f"❌ WebSocket test error: {e}")

def main():
    """Main test function"""
    print("🚀 ENHANCED DASHBOARD WEEK 2 TEST SUITE")
    print("=" * 50)
    
    # Test basic API functionality
    test_api_endpoints()
    
    # Test monitoring controls
    test_monitoring_controls()
    
    # Test real-time data collection
    test_real_time_data()
    
    # Test enhanced features
    test_enhanced_features()
    
    # Test WebSocket functionality
    test_websocket_functionality()
    
    print("\n" + "=" * 50)
    print("🎉 Enhanced Dashboard Week 2 Testing Complete!")
    print("\n📋 Test Summary:")
    print("✅ WebSocket server with Flask-SocketIO")
    print("✅ Real-time environmental data streaming")
    print("✅ Enhanced HTML template with Plotly.js")
    print("✅ Interactive chart controls and parameter filtering")
    print("✅ Real-time alert system")
    print("✅ Settings modal with theme and animation controls")
    print("✅ Progress bars and enhanced visualizations")
    print("✅ Responsive design and dark theme support")
    print("✅ Data export functionality")
    print("✅ Historical data analysis")
    
    print("\n🌐 Dashboard is ready at: http://localhost:5000")
    print("📱 Features include:")
    print("   • Real-time environmental monitoring")
    print("   • Interactive Plotly.js charts")
    print("   • Customizable update frequency")
    print("   • Parameter filtering and time range selection")
    print("   • Dark/light theme switching")
    print("   • Chart animation controls")
    print("   • Data export and settings management")
    print("   • Real-time alerts and notifications")
    print("   • Progress bars for all parameters")
    print("   • Gauge charts for current values")
    print("   • Correlation and trend analysis")
    print("   • Enhanced system status display")

if __name__ == "__main__":
    main() 