#!/usr/bin/env python3
"""
Test Script for Week 2 Web Dashboard
Verifies all components are working correctly
"""

import os
import sys
import json
from pathlib import Path

def test_flask_app():
    """Test Flask app import and basic functionality"""
    print("🧪 Testing Flask App...")
    
    try:
        import app
        print("✅ Flask app imported successfully")
        
        # Test basic app properties
        if hasattr(app, 'app') and hasattr(app, 'socketio'):
            print("✅ Flask app and SocketIO initialized")
        else:
            print("❌ Missing Flask app or SocketIO")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Flask app import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def test_templates():
    """Test HTML template existence"""
    print("\n🧪 Testing Templates...")
    
    template_path = Path("templates/dashboard.html")
    if template_path.exists():
        print("✅ Dashboard template exists")
        return True
    else:
        print("❌ Dashboard template missing")
        return False

def test_static_files():
    """Test static files existence"""
    print("\n🧪 Testing Static Files...")
    
    static_files = [
        "static/css/dashboard.css",
        "static/js/dashboard.js", 
        "static/js/websocket.js",
        "static/js/charts.js"
    ]
    
    all_exist = True
    for file_path in static_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_directory_structure():
    """Test web directory structure"""
    print("\n🧪 Testing Directory Structure...")
    
    required_dirs = ["templates", "static", "static/css", "static/js"]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/ directory exists")
        else:
            print(f"❌ {dir_path}/ directory missing")
            all_exist = False
    
    return all_exist

def test_virtual_environment():
    """Test virtual environment activation"""
    print("\n🧪 Testing Virtual Environment...")
    
    try:
        import flask
        import flask_socketio
        print("✅ Flask and Flask-SocketIO available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Activate virtual environment: source ../phase3_venv/bin/activate")
        return False

def main():
    """Run all tests"""
    print("🚀 Week 2 Web Dashboard Test Suite")
    print("=" * 50)
    
    tests = [
        test_virtual_environment,
        test_directory_structure,
        test_templates,
        test_static_files,
        test_flask_app
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Web dashboard is ready for deployment.")
        print("\n🚀 To start the web server:")
        print("   python3 app.py")
        print("\n🌐 Access dashboard at: http://localhost:5000")
    else:
        print("\n🔧 Some tests failed. Please fix issues before deployment.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 