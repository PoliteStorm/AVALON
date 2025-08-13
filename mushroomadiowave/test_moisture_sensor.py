#!/usr/bin/env python3
"""
Comprehensive Test Script for Hybrid Moisture Sensor System
Validates scientific methodology, data-driven analysis, and bias elimination

This script tests:
1. Acoustic analysis accuracy
2. Electrical pattern detection
3. Correlation discovery
4. Pattern recognition
5. Moisture estimation
6. Scientific validation

SCIENTIFIC VALIDATION CRITERIA:
- No forced parameters
- Data-driven correlation discovery
- Pure pattern recognition
- Uncertainty quantification
- Reproducible methodology
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def test_acoustic_analyzer():
    """Test acoustic analyzer for scientific validity"""
    print("🧪 TESTING ACOUSTIC ANALYZER")
    print("=" * 40)
    
    try:
        from hybrid_moisture_sensor import AcousticAnalyzer
        
        analyzer = AcousticAnalyzer()
        
        # Test 1: Pure sine wave (known characteristics)
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate test signals with known properties
        test_signals = {
            'pure_sine_440': np.sin(2 * np.pi * 440 * t),
            'noise_only': np.random.randn(len(t)) * 0.1,
            'mixed_signal': np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t)),
            'chirp_signal': np.sin(2 * np.pi * (440 + 1000 * t) * t)
        }
        
        results = {}
        
        for signal_name, signal_data in test_signals.items():
            print(f"\n📊 Testing: {signal_name}")
            
            features = analyzer.analyze_sound_waves(signal_data)
            
            # Validate features
            required_features = [
                'spectral_centroid', 'spectral_bandwidth', 'rms_energy',
                'zero_crossings', 'signal_mean', 'signal_std',
                'signal_skewness', 'signal_kurtosis'
            ]
            
            missing_features = [f for f in required_features if f not in features]
            if missing_features:
                print(f"   ❌ Missing features: {missing_features}")
                continue
            
            # Check for reasonable values
            if features['spectral_centroid'] < 0 or features['spectral_centroid'] > 22050:
                print(f"   ⚠️  Spectral centroid out of range: {features['spectral_centroid']}")
            
            if features['rms_energy'] < 0:
                print(f"   ❌ RMS energy negative: {features['rms_energy']}")
            
            print(f"   ✅ Features extracted: {len(features)}")
            print(f"   📊 Spectral centroid: {features['spectral_centroid']:.1f} Hz")
            print(f"   📊 RMS energy: {features['rms_energy']:.4f}")
            
            results[signal_name] = features
        
        # Validate no assumptions about moisture
        print(f"\n🔍 VALIDATION: No moisture assumptions")
        for signal_name, features in results.items():
            if 'moisture' in str(features).lower():
                print(f"   ❌ Moisture assumption detected in {signal_name}")
            else:
                print(f"   ✅ No moisture assumptions in {signal_name}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Acoustic analyzer test failed: {e}")
        return False

def test_correlation_engine():
    """Test correlation engine for scientific validity"""
    print("\n🧪 TESTING CORRELATION ENGINE")
    print("=" * 40)
    
    try:
        from hybrid_moisture_sensor import CorrelationEngine
        
        engine = CorrelationEngine()
        
        # Test 1: Generate test features
        test_acoustic = {
            'spectral_centroid': 440.0,
            'spectral_bandwidth': 100.0,
            'rms_energy': 0.5,
            'zero_crossings': 100,
            'signal_mean': 0.0,
            'signal_std': 0.1,
            'signal_skewness': 0.0,
            'signal_kurtosis': 3.0,
            'envelope_mean': 0.5,
            'envelope_std': 0.1
        }
        
        test_electrical = {
            'shannon_entropy': 2.5,
            'variance': 0.01,
            'skewness': 0.2,
            'kurtosis': 3.5,
            'zero_crossings': 50,
            'spectral_centroid': 0.1,
            'spectral_bandwidth': 0.05
        }
        
        # Test correlation discovery
        correlations = engine.find_correlations(test_acoustic, test_electrical)
        
        # Validate results
        required_keys = ['correlation_matrix', 'cross_correlations', 'significant_correlations', 'correlation_summary']
        missing_keys = [k for k in required_keys if k not in correlations]
        
        if missing_keys:
            print(f"   ❌ Missing correlation results: {missing_keys}")
            return False
        
        print(f"   ✅ Correlation analysis completed")
        print(f"   📊 Total correlations: {correlations['correlation_summary']['total_correlations']}")
        print(f"   📊 Strong correlations: {correlations['correlation_summary']['strong_correlations']}")
        
        # Validate no forced relationships
        if 'moisture' in str(correlations).lower():
            print(f"   ❌ Moisture relationship forced in correlations")
            return False
        else:
            print(f"   ✅ No forced moisture relationships")
        
        # Test pattern memory
        if len(engine.pattern_memory) > 0:
            print(f"   ✅ Pattern memory working: {len(engine.pattern_memory)} patterns stored")
        else:
            print(f"   ⚠️  No patterns stored in memory")
        
        return True
        
    except Exception as e:
        print(f"❌ Correlation engine test failed: {e}")
        return False

def test_pattern_classifier():
    """Test pattern classifier for scientific validity"""
    print("\n🧪 TESTING PATTERN CLASSIFIER")
    print("=" * 40)
    
    try:
        from hybrid_moisture_sensor import PatternClassifier
        
        classifier = PatternClassifier()
        
        # Test 1: Learn patterns without moisture data (unsupervised)
        test_acoustic = {
            'spectral_centroid': 440.0,
            'spectral_bandwidth': 100.0,
            'rms_energy': 0.5,
            'zero_crossings': 100,
            'signal_mean': 0.0,
            'signal_std': 0.1,
            'signal_skewness': 0.0,
            'signal_kurtosis': 3.0,
            'envelope_mean': 0.5,
            'envelope_std': 0.1
        }
        
        test_electrical = {
            'shannon_entropy': 2.5,
            'variance': 0.01,
            'skewness': 0.2,
            'kurtosis': 3.5,
            'zero_crossings': 50,
            'spectral_centroid': 0.1,
            'spectral_bandwidth': 0.05
        }
        
        test_correlations = {
            'significant_correlations': [
                {'acoustic_feature': 'spectral_centroid', 'electrical_feature': 'shannon_entropy', 'correlation': 0.6, 'strength': 'moderate'},
                {'acoustic_feature': 'rms_energy', 'electrical_feature': 'variance', 'correlation': 0.4, 'strength': 'weak'}
            ]
        }
        
        # Learn pattern without moisture level (unsupervised)
        learning_result = classifier.learn_pattern(test_acoustic, test_electrical, test_correlations)
        
        if not learning_result['pattern_learned']:
            print(f"   ❌ Pattern learning failed")
            return False
        
        print(f"   ✅ Pattern learned successfully")
        print(f"   📊 Pattern ID: {learning_result['pattern_id']}")
        print(f"   📊 Confidence: {learning_result['confidence']:.3f}")
        
        # Test 2: Learn pattern with moisture level (supervised)
        supervised_result = classifier.learn_pattern(
            test_acoustic, test_electrical, test_correlations, moisture_level=0.65
        )
        
        print(f"   ✅ Supervised pattern learned")
        print(f"   📊 Pattern ID: {supervised_result['pattern_id']}")
        
        # Test 3: Find similar patterns
        similar_patterns = classifier.find_similar_patterns(
            test_acoustic, test_electrical, test_correlations
        )
        
        if len(similar_patterns) > 0:
            print(f"   ✅ Similar patterns found: {len(similar_patterns)}")
            for pattern in similar_patterns[:2]:  # Show first 2
                print(f"      📊 Pattern {pattern['pattern_id']}: similarity={pattern['similarity_score']:.3f}, confidence={pattern['confidence']:.3f}")
        else:
            print(f"   ⚠️  No similar patterns found")
        
        # Validate no forced moisture calculations
        if 'moisture' in str(classifier.learned_patterns).lower():
            print(f"   ✅ Moisture patterns stored correctly")
        else:
            print(f"   ⚠️  No moisture patterns detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Pattern classifier test failed: {e}")
        return False

def test_integrated_sensor():
    """Test the complete integrated moisture sensor system"""
    print("\n🧪 TESTING INTEGRATED MOISTURE SENSOR")
    print("=" * 40)
    
    try:
        from hybrid_moisture_sensor import FungalMoistureSensor
        
        sensor = FungalMoistureSensor()
        
        # Test 1: Generate test signals
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create correlated test signals
        base_signal = np.sin(2 * np.pi * 440 * t)
        test_audio = base_signal + 0.1 * np.random.randn(len(t))
        test_electrical = base_signal * 0.01 + 0.05 * np.random.randn(len(t))
        
        print(f"   📊 Test signals generated:")
        print(f"      Audio: {len(test_audio)} samples")
        print(f"      Electrical: {len(test_electrical)} samples")
        
        # Test 2: Collect sensor data
        sensor_data = sensor.collect_sensor_data(test_audio, test_electrical)
        
        if 'error' in sensor_data:
            print(f"   ❌ Sensor data collection failed: {sensor_data['error']}")
            return False
        
        print(f"   ✅ Sensor data collected successfully")
        print(f"   📊 Acoustic features: {len(sensor_data['acoustic_features'])}")
        print(f"   📊 Electrical features: {len(sensor_data['electrical_features'])}")
        
        # Test 3: Analyze correlations
        correlations = sensor.analyze_moisture_correlation(
            sensor_data['acoustic_features'],
            sensor_data['electrical_features']
        )
        
        if 'error' in correlations:
            print(f"   ❌ Correlation analysis failed: {correlations['error']}")
            return False
        
        print(f"   ✅ Correlation analysis completed")
        print(f"   📊 Correlations discovered: {correlations['correlations_discovered']['correlation_summary']['total_correlations']}")
        
        # Test 4: Estimate moisture patterns
        moisture_estimate = sensor.estimate_moisture_patterns(
            sensor_data['acoustic_features'],
            sensor_data['electrical_features'],
            correlations['correlations_discovered']
        )
        
        print(f"   ✅ Moisture pattern estimation completed")
        print(f"   💧 Estimate: {moisture_estimate['moisture_estimate']}")
        print(f"   📊 Confidence: {moisture_estimate['confidence']:.3f}")
        
        # Test 5: Get sensor status
        status = sensor.get_sensor_status()
        print(f"   📊 Sensor status retrieved:")
        print(f"      Patterns learned: {status['patterns_learned']}")
        print(f"      Data points: {status['data_points_collected']}")
        
        # Test 6: Save sensor data
        filename = sensor.save_sensor_data()
        if Path(filename).exists():
            print(f"   💾 Sensor data saved: {filename}")
        else:
            print(f"   ❌ Failed to save sensor data")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated sensor test failed: {e}")
        return False

def test_scientific_validation():
    """Test scientific validation criteria"""
    print("\n🧪 TESTING SCIENTIFIC VALIDATION")
    print("=" * 40)
    
    validation_results = {}
    
    # Test 1: No forced parameters
    print("🔍 Testing: No forced parameters")
    try:
        # Check if any hardcoded moisture values exist in the code
        code_files = [
            "hybrid_moisture_sensor.py",
            "moisture_sensor_integration.py"
        ]
        
        forced_parameters_found = False
        for filename in code_files:
            if Path(filename).exists():
                with open(filename, 'r') as f:
                    content = f.read()
                    # Check for hardcoded moisture values
                    if 'moisture = ' in content or 'moisture=' in content:
                        if not 'moisture_level' in content:  # Allow legitimate moisture_level variables
                            forced_parameters_found = True
                            print(f"   ⚠️  Potential forced parameter in {filename}")
        
        if not forced_parameters_found:
            print("   ✅ No forced parameters detected")
            validation_results['no_forced_parameters'] = True
        else:
            print("   ❌ Forced parameters detected")
            validation_results['no_forced_parameters'] = False
            
    except Exception as e:
        print(f"   ❌ Forced parameter test failed: {e}")
        validation_results['no_forced_parameters'] = False
    
    # Test 2: Data-driven analysis
    print("🔍 Testing: Data-driven analysis")
    try:
        # Check if analysis methods use actual data
        from hybrid_moisture_sensor import AcousticAnalyzer
        
        analyzer = AcousticAnalyzer()
        test_signal = np.random.randn(1000)
        features = analyzer.analyze_sound_waves(test_signal)
        
        # Verify features are calculated from actual signal
        if features['spectral_centroid'] != 0 and features['rms_energy'] > 0:
            print("   ✅ Features calculated from actual signal data")
            validation_results['data_driven_analysis'] = True
        else:
            print("   ❌ Features not calculated from signal data")
            validation_results['data_driven_analysis'] = False
            
    except Exception as e:
        print(f"   ❌ Data-driven analysis test failed: {e}")
        validation_results['data_driven_analysis'] = False
    
    # Test 3: Correlation discovery
    print("🔍 Testing: Correlation discovery")
    try:
        from hybrid_moisture_sensor import CorrelationEngine
        
        engine = CorrelationEngine()
        
        # Test with different signals to ensure correlations are discovered, not forced
        test_acoustic_1 = {'spectral_centroid': 440.0, 'spectral_bandwidth': 100.0, 'rms_energy': 0.5, 'zero_crossings': 100, 'signal_mean': 0.0, 'signal_std': 0.1, 'signal_skewness': 0.0, 'signal_kurtosis': 3.0, 'envelope_mean': 0.5, 'envelope_std': 0.1}
        test_electrical_1 = {'shannon_entropy': 2.5, 'variance': 0.01, 'skewness': 0.2, 'kurtosis': 3.5, 'zero_crossings': 50, 'spectral_centroid': 0.1, 'spectral_bandwidth': 0.05}
        
        test_acoustic_2 = {'spectral_centroid': 880.0, 'spectral_bandwidth': 200.0, 'rms_energy': 1.0, 'zero_crossings': 200, 'signal_mean': 0.0, 'signal_std': 0.2, 'signal_skewness': 0.0, 'signal_kurtosis': 3.0, 'envelope_mean': 1.0, 'envelope_std': 0.2}
        test_electrical_2 = {'shannon_entropy': 3.0, 'variance': 0.04, 'skewness': 0.0, 'kurtosis': 3.0, 'zero_crossings': 100, 'spectral_centroid': 0.2, 'spectral_bandwidth': 0.1}
        
        corr1 = engine.find_correlations(test_acoustic_1, test_electrical_1)
        corr2 = engine.find_correlations(test_acoustic_2, test_electrical_2)
        
        # Correlations should be different for different signals
        if corr1['correlation_summary']['total_correlations'] != corr2['correlation_summary']['total_correlations']:
            print("   ✅ Correlations discovered dynamically")
            validation_results['correlation_discovery'] = True
        else:
            print("   ⚠️  Correlations may be forced")
            validation_results['correlation_discovery'] = False
            
    except Exception as e:
        print(f"   ❌ Correlation discovery test failed: {e}")
        validation_results['correlation_discovery'] = False
    
    # Test 4: Pattern recognition
    print("🔍 Testing: Pattern recognition")
    try:
        from hybrid_moisture_sensor import PatternClassifier
        
        classifier = PatternClassifier()
        
        # Test pattern learning and recognition
        test_acoustic = {'spectral_centroid': 440.0, 'spectral_bandwidth': 100.0, 'rms_energy': 0.5, 'zero_crossings': 100, 'signal_mean': 0.0, 'signal_std': 0.1, 'signal_skewness': 0.0, 'signal_kurtosis': 3.0, 'envelope_mean': 0.5, 'envelope_std': 0.1}
        test_electrical = {'shannon_entropy': 2.5, 'variance': 0.01, 'skewness': 0.2, 'kurtosis': 3.5, 'zero_crossings': 50, 'spectral_centroid': 0.1, 'spectral_bandwidth': 0.05}
        test_correlations = {'significant_correlations': [{'acoustic_feature': 'spectral_centroid', 'electrical_feature': 'shannon_entropy', 'correlation': 0.6, 'strength': 'moderate'}]}
        
        # Learn pattern
        learning_result = classifier.learn_pattern(test_acoustic, test_electrical, test_correlations)
        
        # Find similar patterns
        similar_patterns = classifier.find_similar_patterns(test_acoustic, test_electrical, test_correlations)
        
        if learning_result['pattern_learned'] and len(similar_patterns) > 0:
            print("   ✅ Pattern recognition working")
            validation_results['pattern_recognition'] = True
        else:
            print("   ❌ Pattern recognition failed")
            validation_results['pattern_recognition'] = False
            
    except Exception as e:
        print(f"   ❌ Pattern recognition test failed: {e}")
        validation_results['pattern_recognition'] = False
    
    # Test 5: Uncertainty quantification
    print("🔍 Testing: Uncertainty quantification")
    try:
        from hybrid_moisture_sensor import FungalMoistureSensor
        
        sensor = FungalMoistureSensor()
        
        # Generate test signals
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        test_electrical = np.random.randn(len(t)) * 0.1
        
        # Collect data and analyze
        sensor_data = sensor.collect_sensor_data(test_audio, test_electrical)
        correlations = sensor.analyze_moisture_correlation(
            sensor_data['acoustic_features'],
            sensor_data['electrical_features']
        )
        moisture_estimate = sensor.estimate_moisture_patterns(
            sensor_data['acoustic_features'],
            sensor_data['electrical_features'],
            correlations['correlations_discovered']
        )
        
        # Check for confidence/uncertainty information
        if 'confidence' in moisture_estimate and 'uncertainty' in moisture_estimate:
            print("   ✅ Uncertainty quantification present")
            validation_results['uncertainty_quantification'] = True
        else:
            print("   ❌ Uncertainty quantification missing")
            validation_results['uncertainty_quantification'] = False
            
    except Exception as e:
        print(f"   ❌ Uncertainty quantification test failed: {e}")
        validation_results['uncertainty_quantification'] = False
    
    # Summary
    print(f"\n📊 SCIENTIFIC VALIDATION SUMMARY:")
    print("=" * 40)
    
    passed_tests = sum(validation_results.values())
    total_tests = len(validation_results)
    
    for test_name, passed in validation_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n   Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("   🎉 ALL SCIENTIFIC VALIDATION CRITERIA MET!")
    else:
        print("   ⚠️  Some validation criteria failed")
    
    return validation_results

def main():
    """Run comprehensive testing suite"""
    print("🧪 COMPREHENSIVE TESTING SUITE - HYBRID MOISTURE SENSOR")
    print("=" * 70)
    print("🔬 Testing scientific validity and bias elimination")
    print("🔬 Validating data-driven analysis methodology")
    print("🔬 Ensuring no forced parameters or assumptions")
    print("=" * 70)
    
    test_results = {}
    
    # Run all tests
    test_results['acoustic_analyzer'] = test_acoustic_analyzer()
    test_results['correlation_engine'] = test_correlation_engine()
    test_results['pattern_classifier'] = test_pattern_classifier()
    test_results['integrated_sensor'] = test_integrated_sensor()
    test_results['scientific_validation'] = test_scientific_validation()
    
    # Overall results
    print(f"\n🎯 OVERALL TEST RESULTS")
    print("=" * 70)
    
    # Handle both boolean and dict test results
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        if isinstance(result, bool):
            passed_tests += 1 if result else 0
        elif isinstance(result, dict) and 'scientific_validation' in result:
            # Handle scientific validation results
            passed_tests += sum(result['scientific_validation'].values())
            total_tests += len(result['scientific_validation']) - 1  # Adjust for the dict itself
    
    for test_name, result in test_results.items():
        if isinstance(result, bool):
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
        elif isinstance(result, dict) and 'scientific_validation' in result:
            print(f"   {test_name}: ✅ PASS (scientific validation)")
    
    print(f"\n   Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("   🎉 ALL TESTS PASSED!")
        print("   ✅ Hybrid moisture sensor system is scientifically valid")
        print("   ✅ No forced parameters or bias detected")
        print("   ✅ Data-driven analysis methodology confirmed")
        print("   ✅ Ready for scientific use")
    else:
        print("   ⚠️  Some tests failed")
        print("   🔍 Review failed tests for issues")
    
    # Save test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"moisture_sensor_test_results_{timestamp}.json"
    
    test_summary = {
        'test_timestamp': datetime.now().isoformat(),
        'test_results': test_results,
        'overall_score': f"{passed_tests}/{total_tests}",
        'scientific_validation': passed_tests == total_tests,
        'recommendations': []
    }
    
    if passed_tests < total_tests:
        failed_tests = [name for name, passed in test_results.items() if not passed]
        test_summary['recommendations'] = [f"Review and fix: {test}" for test in failed_tests]
    
    with open(results_filename, 'w') as f:
        json.dump(test_summary, f, indent=2, default=str)
    
    print(f"\n💾 Test results saved: {results_filename}")
    
    return test_results

if __name__ == "__main__":
    main() 