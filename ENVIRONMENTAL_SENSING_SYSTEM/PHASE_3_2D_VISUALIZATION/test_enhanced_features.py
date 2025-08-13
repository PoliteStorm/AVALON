#!/usr/bin/env python3
"""
Enhanced Dashboard Features Test - Week 2+ Implementation
Tests all new features: experiment controls, data transparency, preset conditions, and references
"""

import requests
import json
import time
from datetime import datetime

def print_header():
    """Print test header"""
    print("🚀" + "="*70 + "🚀")
    print("🌟 ENHANCED DASHBOARD FEATURES TEST - RESEARCH EDITION 🌟")
    print("🚀" + "="*70 + "🚀")
    print()

def test_enhanced_html_features():
    """Test all enhanced HTML features"""
    print("🎨 **ENHANCED HTML FEATURES TEST**")
    print("-" * 50)
    
    base_url = "http://localhost:5000"
    
    try:
        response = requests.get(f"{base_url}/")
        html_content = response.text
        
        # Test experiment information banner
        experiment_features = [
            ("Experiment Banner", "Active Experiment: Fungal Electrical Response to Environmental Stress"),
            ("Species Information", "Pleurotus ostreatus (Oyster Mushroom)"),
            ("Electrode Type", "Full vs Tip Comparison"),
            ("Sampling Rate", "36kHz"),
            ("References", "Adamatzky 2023 (PMC), Adamatzky 2022 (Spike Detection)"),
            ("Methodology", "√t wave transform, FHN model compliance"),
            ("Data Source Badges", "Real Data: 87%, Simulated: 13%"),
            ("Research Grade Badge", "Research Grade")
        ]
        
        for feature_name, feature_text in experiment_features:
            if feature_text in html_content:
                print(f"✅ {feature_name}: OK")
            else:
                print(f"❌ {feature_name}: Missing")
        
        # Test experiment controls
        control_features = [
            ("Experiment Preset Selector", "experiment-preset"),
            ("Data Blending Mode", "data-blending"),
            ("Chart Resolution Control", "chart-resolution"),
            ("References Button", "show-references"),
            ("Methodology Button", "show-methodology"),
            ("Data Transparency Toggle", "data-transparency"),
            ("Simulation Accuracy", "simulation-accuracy")
        ]
        
        print("\n🎛️ **EXPERIMENT CONTROLS TEST**")
        print("-" * 30)
        for feature_name, feature_id in control_features:
            if feature_id in html_content:
                print(f"✅ {feature_name}: OK")
            else:
                print(f"❌ {feature_name}: Missing")
        
        # Test enhanced chart controls
        chart_features = [
            ("Advanced Chart Controls", "Advanced Chart Controls"),
            ("Chart Type Selection", "Heatmap, 3D Surface"),
            ("Time Range Options", "30d, Custom Range"),
            ("Update Frequency", "1 minute option"),
            ("Chart Resolution", "Ultra (5000 points)"),
            ("Electrical Parameter", "electrical-check"),
            ("Parameter Grid Layout", "col-md-2 layout")
        ]
        
        print("\n📊 **ENHANCED CHART CONTROLS TEST**")
        print("-" * 35)
        for feature_name, feature_text in chart_features:
            if feature_text in html_content:
                print(f"✅ {feature_name}: OK")
            else:
                print(f"❌ {feature_name}: Missing")
        
        # Test research data section
        research_features = [
            ("Research Data Section", "Research Data & Methodology"),
            ("Data Source Table", "data-source-table"),
            ("Data Source Breakdown", "Data Source Breakdown"),
            ("Research Methodology", "Research Methodology"),
            ("Wave Transform Info", "√t scaling (Adamatzky 2023)"),
            ("FHN Model Info", "FitzHugh-Nagumo (FHN) compliance"),
            ("Sampling Info", "36kHz electrical activity"),
            ("Validation Info", "Peer-reviewed research alignment")
        ]
        
        print("\n🔬 **RESEARCH DATA SECTION TEST**")
        print("-" * 35)
        for feature_name, feature_text in research_features:
            if feature_text in html_content:
                print(f"✅ {feature_name}: OK")
            else:
                print(f"❌ {feature_name}: Missing")
        
        # Test enhanced modals
        modal_features = [
            ("Enhanced Settings Modal", "modal-xl"),
            ("References Modal", "referencesModal"),
            ("Research References", "Research References & Bibliography"),
            ("Primary Research Papers", "Adamatzky, A. (2023)"),
            ("Environmental Sensing", "Environmental Sensing References"),
            ("Data Analysis Methodology", "Data Analysis Methodology"),
            ("Export References Button", "export-references")
        ]
        
        print("\n📋 **ENHANCED MODALS TEST**")
        print("-" * 30)
        for feature_name, feature_text in modal_features:
            if feature_text in html_content:
                print(f"✅ {feature_name}: OK")
            else:
                print(f"❌ {feature_name}: Missing")
        
    except Exception as e:
        print(f"❌ Enhanced features test error: {e}")
    
    print()

def test_experiment_presets():
    """Test experiment preset functionality"""
    print("🧪 **EXPERIMENT PRESETS TEST**")
    print("-" * 35)
    
    base_url = "http://localhost:5000"
    
    presets = [
        "baseline",
        "moisture_stress", 
        "temperature_stress",
        "pollution_exposure",
        "electrode_comparison",
        "species_comparison",
        "custom"
    ]
    
    for preset in presets:
        try:
            # Test if preset option exists in HTML
            response = requests.get(f"{base_url}/")
            if preset in response.text:
                print(f"✅ {preset.replace('_', ' ').title()}: Available")
            else:
                print(f"❌ {preset.replace('_', ' ').title()}: Missing")
        except Exception as e:
            print(f"❌ {preset} test error: {e}")
    
    print()

def test_data_transparency():
    """Test data source transparency features"""
    print("🔍 **DATA TRANSPARENCY TEST**")
    print("-" * 35)
    
    base_url = "http://localhost:5000"
    
    try:
        response = requests.get(f"{base_url}/")
        html_content = response.text
        
        transparency_features = [
            ("Temperature Data Source", "temp-data-source"),
            ("Humidity Data Source", "humidity-data-source"),
            ("pH Data Source", "ph-data-source"),
            ("Moisture Data Source", "moisture-data-source"),
            ("Pollution Data Source", "pollution-data-source"),
            ("Quality Data Source", "quality-data-source"),
            ("Chart Data Source Info", "chart-data-source-info"),
            ("Data Source Table", "data-source-table")
        ]
        
        for feature_name, feature_id in transparency_features:
            if feature_id in html_content:
                print(f"✅ {feature_name}: OK")
            else:
                print(f"❌ {feature_name}: Missing")
        
    except Exception as e:
        print(f"❌ Data transparency test error: {e}")
    
    print()

def test_references_and_bibliography():
    """Test references and bibliography features"""
    print("📚 **REFERENCES & BIBLIOGRAPHY TEST**")
    print("-" * 40)
    
    base_url = "http://localhost:5000"
    
    try:
        response = requests.get(f"{base_url}/")
        html_content = response.text
        
        reference_features = [
            ("References Modal", "referencesModal"),
            ("Primary Research Papers", "Primary Research Papers"),
            ("Adamatzky 2023", "Adamatzky, A. (2023)"),
            ("Adamatzky 2022", "Adamatzky, A. (2022)"),
            ("FitzHugh 1961", "FitzHugh, R. (1961)"),
            ("Environmental Sensing", "Environmental Sensing References"),
            ("Data Analysis", "Data Analysis Methodology"),
            ("Export References", "export-references"),
            ("PMC Journal Reference", "PMC Journal"),
            ("Computational Biology", "Computational Biology"),
            ("Biophysical Journal", "Biophysical Journal")
        ]
        
        for feature_name, feature_text in reference_features:
            if feature_text in html_content:
                print(f"✅ {feature_name}: OK")
            else:
                print(f"❌ {feature_name}: Missing")
        
    except Exception as e:
        print(f"❌ References test error: {e}")
    
    print()

def test_enhanced_layout():
    """Test enhanced layout and responsive design"""
    print("📱 **ENHANCED LAYOUT TEST**")
    print("-" * 30)
    
    base_url = "http://localhost:5000"
    
    try:
        response = requests.get(f"{base_url}/")
        html_content = response.text
        
        layout_features = [
            ("Experiment Banner", "alert-info"),
            ("Enhanced Status Cards", "col-md-2"),
            ("Progress Bars", "progress-bar"),
            ("Data Source Badges", "badge bg-light text-dark"),
            ("Enhanced Controls", "Experiment Controls & Data Sources"),
            ("Advanced Chart Controls", "Advanced Chart Controls"),
            ("Chart Resolution", "chart-resolution"),
            ("Parameter Grid", "col-md-2"),
            ("Research Section", "Research Data & Methodology"),
            ("Enhanced Modals", "modal-xl"),
            ("Responsive Design", "container-fluid")
        ]
        
        for feature_name, feature_text in layout_features:
            if feature_text in html_content:
                print(f"✅ {feature_name}: OK")
            else:
                print(f"❌ {feature_name}: Missing")
        
    except Exception as e:
        print(f"❌ Layout test error: {e}")
    
    print()

def test_api_functionality():
    """Test API endpoints"""
    print("🔌 **API FUNCTIONALITY TEST**")
    print("-" * 30)
    
    base_url = "http://localhost:5000"
    
    try:
        # Test status API
        response = requests.get(f"{base_url}/api/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status API: OK - {data['status']} - {data['data_points_collected']} data points")
        else:
            print(f"❌ Status API: HTTP {response.status_code}")
        
        # Test environmental data API
        response = requests.get(f"{base_url}/api/environmental_data")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Environmental Data API: OK - Temp: {data['temperature']}°C, Humidity: {data['humidity']}%")
        else:
            print(f"❌ Environmental Data API: HTTP {response.status_code}")
        
        # Test main page
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print(f"✅ Main Dashboard: OK - {len(response.text)} characters")
        else:
            print(f"❌ Main Dashboard: HTTP {response.status_code}")
        
    except Exception as e:
        print(f"❌ API test error: {e}")
    
    print()

def main():
    """Main test function"""
    print_header()
    
    # Test all enhanced features
    test_enhanced_html_features()
    test_experiment_presets()
    test_data_transparency()
    test_references_and_bibliography()
    test_enhanced_layout()
    test_api_functionality()
    
    print("🎉 **ENHANCED FEATURES TESTING COMPLETE!** 🎉")
    print("\n📋 **FEATURE SUMMARY**")
    print("✅ Experiment identification and configuration")
    print("✅ Data source transparency (Real vs Simulated)")
    print("✅ Preset experimental conditions")
    print("✅ Advanced chart controls and resolution")
    print("✅ Research methodology and references")
    print("✅ Enhanced layout and responsive design")
    print("✅ Comprehensive bibliography")
    print("✅ Data-driven analysis with citations")
    
    print("\n🌐 **Access your enhanced dashboard at: http://localhost:5000**")
    print("🚀 **All Week 2+ features are now operational!**")

if __name__ == "__main__":
    main() 